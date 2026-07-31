<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

use PapiAI\Anthropic\AnthropicProvider;
use PapiAI\Core\Effort;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Message;

/**
 * Captures the request payload so effort mapping can be asserted without HTTP.
 */
class TestableAnthropicEffortProvider extends AnthropicProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['content' => [['type' => 'text', 'text' => 'ok']], 'stop_reason' => 'end_turn'];
    }
}

describe('AnthropicProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableAnthropicEffortProvider('test-api-key');
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
    });

    it('turns a level into extended thinking with a token budget', function () {
        ($this->chat)(['effort' => 'medium', 'maxTokens' => 20_000]);

        expect($this->provider->lastPayload['thinking']['type'])->toBe('enabled');
        expect($this->provider->lastPayload['thinking']['budget_tokens'])->toBeInt();
    });

    it('spends more of the ceiling as effort rises', function () {
        $budgets = [];

        foreach (['low', 'medium', 'high'] as $level) {
            ($this->chat)(['effort' => $level, 'maxTokens' => 20_000]);
            $budgets[] = $this->provider->lastPayload['thinking']['budget_tokens'];
        }

        expect($budgets[0])->toBeLessThan($budgets[1]);
        expect($budgets[1])->toBeLessThan($budgets[2]);
    });

    it('keeps the budget inside the window the API enforces', function () {
        // Anthropic rejects a budget under 1024, or one that leaves no room to answer.
        foreach (['low', 'medium', 'high'] as $level) {
            ($this->chat)(['effort' => $level, 'maxTokens' => 4_096]);
            $budget = $this->provider->lastPayload['thinking']['budget_tokens'];

            expect($budget)->toBeGreaterThanOrEqual(1_024);
            expect($budget)->toBeLessThan(4_096);
        }
    });

    it('refuses a ceiling too small to both think and answer', function () {
        // Silently dropping the option would be worse: the caller asked to think and would be
        // billed for a shallow answer instead.
        expect(fn () => ($this->chat)(['effort' => 'low', 'maxTokens' => 1_200]))
            ->toThrow(ProviderException::class, 'maxTokens');

        expect($this->provider->lastPayload)->toBe([]);
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)([]);

        expect($this->provider->lastPayload)->not->toHaveKey('thinking');
    });

    it('rejects a level it does not recognise', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');
    });

    it('sends no thinking block at all for none', function () {
        ($this->chat)(['effort' => 'none', 'maxTokens' => 20_000]);

        expect($this->provider->lastPayload)->not->toHaveKey('thinking');
    });

    it('honours the whole scale, since its knob is continuous', function () {
        $budgets = [];

        foreach (['minimal', 'low', 'medium', 'high', 'extra-high', 'maximum'] as $level) {
            ($this->chat)(['effort' => $level, 'maxTokens' => 100_000]);
            $budgets[] = $this->provider->lastPayload['thinking']['budget_tokens'];
        }

        expect($budgets)->toBe(array_unique($budgets));
        expect($budgets)->toBe(array_values(array_filter($budgets, fn ($b) => $b > 0)));
    });

    it('accepts a provider-level default the call can override', function () {
        $provider = new TestableAnthropicEffortProvider('k', 'claude-sonnet-4-20250514', 20_000, Effort::High);

        $provider->chat([Message::user('hi')], []);
        $fromDefault = $provider->lastPayload['thinking']['budget_tokens'];

        $provider->chat([Message::user('hi')], ['effort' => 'low']);

        expect($provider->lastPayload['thinking']['budget_tokens'])->toBeLessThan($fromDefault);
    });
});
