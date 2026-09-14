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

    // Every current Claude rejects a token budget with a 400. The knob is adaptive thinking plus
    // an effort level, which is a different shape entirely.
    describe('on the current generation (adaptive thinking)', function () {
        it('sends adaptive thinking and an effort level, never a budget', function () {
            ($this->chat)(['effort' => 'medium']);

            expect($this->provider->lastPayload['thinking'])->toBe(['type' => 'adaptive']);
            expect($this->provider->lastPayload['output_config'])->toBe(['effort' => 'medium']);
            expect($this->provider->lastPayload['thinking'])->not->toHaveKey('budget_tokens');
        });

        it('translates the two levels Anthropic spells differently', function () {
            ($this->chat)(['effort' => 'extra-high']);
            expect($this->provider->lastPayload['output_config']['effort'])->toBe('xhigh');

            ($this->chat)(['effort' => 'maximum']);
            expect($this->provider->lastPayload['output_config']['effort'])->toBe('max');
        });

        it('narrows minimal to low, the shallowest level offered', function () {
            ($this->chat)(['effort' => 'minimal']);

            expect($this->provider->lastPayload['output_config']['effort'])->toBe('low');
        });

        it('has no xhigh on the 4.6 pair, so extra-high rounds up to max', function () {
            ($this->chat)(['effort' => 'extra-high', 'model' => AnthropicProvider::MODEL_CLAUDE_SONNET_4_6]);

            expect($this->provider->lastPayload['output_config']['effort'])->toBe('max');
        });

        it('needs no headroom check, since the API paces itself', function () {
            ($this->chat)(['effort' => 'high', 'maxTokens' => 1_200]);

            expect($this->provider->lastPayload['thinking'])->toBe(['type' => 'adaptive']);
        });

        it('switches thinking off explicitly for none where the model would otherwise think', function () {
            // Sonnet 5 and Opus 5 think when the field is omitted, so "none" has to say so.
            ($this->chat)(['effort' => 'none']);

            expect($this->provider->lastPayload['thinking'])->toBe(['type' => 'disabled']);
            expect($this->provider->lastPayload)->not->toHaveKey('output_config');
        });

        it('omits thinking for none on models that are off by default', function () {
            ($this->chat)(['effort' => 'none', 'model' => AnthropicProvider::MODEL_CLAUDE_OPUS_4_8]);

            expect($this->provider->lastPayload)->not->toHaveKey('thinking');
            expect($this->provider->lastPayload)->not->toHaveKey('output_config');
        });
    });

    // Fable cannot stop thinking. Asking for none narrows to the shallowest effort, the same
    // posture as Gemini 3, rather than sending a disabled block the API rejects.
    describe('on Fable, where thinking is always on', function () {
        it('never sends a thinking block, and maps effort to output_config', function () {
            ($this->chat)(['effort' => 'high', 'model' => AnthropicProvider::MODEL_CLAUDE_FABLE_5]);

            expect($this->provider->lastPayload)->not->toHaveKey('thinking');
            expect($this->provider->lastPayload['output_config'])->toBe(['effort' => 'high']);
        });

        it('narrows none to low instead of pretending to disable', function () {
            ($this->chat)(['effort' => 'none', 'model' => AnthropicProvider::MODEL_CLAUDE_FABLE_5]);

            expect($this->provider->lastPayload)->not->toHaveKey('thinking');
            expect($this->provider->lastPayload['output_config'])->toBe(['effort' => 'low']);
        });
    });

    // Haiku 4.5 predates adaptive thinking: it wants the old budget and rejects an effort level.
    describe('on Haiku 4.5 (token budget)', function () {
        beforeEach(function () {
            $this->haiku = fn (array $options) => ($this->chat)(
                $options + ['model' => AnthropicProvider::MODEL_CLAUDE_HAIKU_4_5],
            );
        });

        it('turns a level into extended thinking with a token budget', function () {
            ($this->haiku)(['effort' => 'medium', 'maxTokens' => 20_000]);

            expect($this->provider->lastPayload['thinking']['type'])->toBe('enabled');
            expect($this->provider->lastPayload['thinking']['budget_tokens'])->toBeInt();
            expect($this->provider->lastPayload)->not->toHaveKey('output_config');
        });

        it('spends more of the ceiling as effort rises', function () {
            $budgets = [];

            foreach (['low', 'medium', 'high'] as $level) {
                ($this->haiku)(['effort' => $level, 'maxTokens' => 20_000]);
                $budgets[] = $this->provider->lastPayload['thinking']['budget_tokens'];
            }

            expect($budgets[0])->toBeLessThan($budgets[1]);
            expect($budgets[1])->toBeLessThan($budgets[2]);
        });

        it('keeps the budget inside the window the API enforces', function () {
            foreach (['low', 'medium', 'high'] as $level) {
                ($this->haiku)(['effort' => $level, 'maxTokens' => 4_096]);
                $budget = $this->provider->lastPayload['thinking']['budget_tokens'];

                expect($budget)->toBeGreaterThanOrEqual(1_024);
                expect($budget)->toBeLessThan(4_096);
            }
        });

        it('refuses a ceiling too small to both think and answer', function () {
            expect(fn () => ($this->haiku)(['effort' => 'low', 'maxTokens' => 1_200]))
                ->toThrow(ProviderException::class, 'maxTokens');

            expect($this->provider->lastPayload)->toBe([]);
        });

        it('sends no thinking block at all for none', function () {
            ($this->haiku)(['effort' => 'none', 'maxTokens' => 20_000]);

            expect($this->provider->lastPayload)->not->toHaveKey('thinking');
        });

        it('honours the whole scale, since its knob is continuous', function () {
            $budgets = [];

            foreach (['minimal', 'low', 'medium', 'high', 'extra-high', 'maximum'] as $level) {
                ($this->haiku)(['effort' => $level, 'maxTokens' => 100_000]);
                $budgets[] = $this->provider->lastPayload['thinking']['budget_tokens'];
            }

            expect($budgets)->toBe(array_unique($budgets));
        });
    });

    it('sends nothing when the caller does not ask, leaving the model to its own default', function () {
        ($this->chat)([]);

        expect($this->provider->lastPayload)->not->toHaveKey('thinking');
        expect($this->provider->lastPayload)->not->toHaveKey('output_config');
    });

    it('rejects a level it does not recognise', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');
    });

    it('accepts a provider-level default the call can override', function () {
        $provider = new TestableAnthropicEffortProvider('k', AnthropicProvider::MODEL_CLAUDE_SONNET_5, 20_000, Effort::High);

        $provider->chat([Message::user('hi')], []);
        expect($provider->lastPayload['output_config']['effort'])->toBe('high');

        $provider->chat([Message::user('hi')], ['effort' => 'low']);
        expect($provider->lastPayload['output_config']['effort'])->toBe('low');
    });
});
