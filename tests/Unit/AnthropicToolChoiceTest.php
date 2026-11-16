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
use PapiAI\Core\Contracts\NamedToolSelectableInterface;
use PapiAI\Core\Contracts\ToolSelectableInterface;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Message;

/**
 * Captures the request payload so tool-choice mapping can be asserted without HTTP.
 */
class TestableAnthropicToolChoiceProvider extends AnthropicProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['content' => [['type' => 'text', 'text' => 'ok']], 'stop_reason' => 'end_turn'];
    }
}

describe('AnthropicProvider tool choice', function () {
    beforeEach(function () {
        // Sonnet 5 accepts a forced choice; the 5.5 generation does not and is covered below.
        $this->provider = new TestableAnthropicToolChoiceProvider('test-api-key', AnthropicProvider::MODEL_CLAUDE_SONNET_5);
        $this->tools = [
            ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object']],
        ];
    });

    it('maps auto to {type: auto}', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'auto']);

        expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'auto']);
    });

    it('maps none to {type: none}', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'none']);

        expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'none']);
    });

    it('maps required to {type: any}', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'required']);

        expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'any']);
    });

    it('maps a specific tool to {type: tool, name}', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => ['name' => 'get_weather']]);

        expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'tool', 'name' => 'get_weather']);
    });

    it('emits no tool_choice when absent (backward compatible)', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools]);

        expect($this->provider->lastPayload)->not->toHaveKey('tool_choice');
    });

    it('throws for required with no tools, before any HTTP call', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['toolChoice' => 'required']))
            ->toThrow(InvalidArgumentException::class);
        expect($this->provider->lastPayload)->toBe([]);
    });

    it('throws for an unknown tool name', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => ['name' => 'nope']]))
            ->toThrow(InvalidArgumentException::class);
    });
});

describe('AnthropicProvider tool-selection capability', function () {
    it('declares what it can force, so callers can ask instead of catching', function () {
        expect(is_subclass_of(AnthropicProvider::class, NamedToolSelectableInterface::class))->toBeTrue();
        expect(is_subclass_of(AnthropicProvider::class, ToolSelectableInterface::class))->toBeTrue();
    });
    // Fable 5.1 returns a 400 for a forced choice, "any" and named alike. Throwing here gives a
    // message that says why, instead of an opaque API error after a round trip.
    describe('on Fable 5.1, which refuses forced choices', function () {
        beforeEach(function () {
            $this->provider = new TestableAnthropicToolChoiceProvider('test-api-key');
            $this->tools = [
                ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object']],
            ];
        });

        it('throws before any HTTP call for a named tool', function () {
            expect(fn () => $this->provider->chat([Message::user('hi')], [
                'tools' => $this->tools,
                'toolChoice' => ['name' => 'get_weather'],
                'model' => AnthropicProvider::MODEL_CLAUDE_FABLE_5_1,
            ]))->toThrow(ProviderException::class, 'forced tool choice');

            expect($this->provider->lastPayload)->toBe([]);
        });

        it('throws for required, which is the same restriction', function () {
            expect(fn () => $this->provider->chat([Message::user('hi')], [
                'tools' => $this->tools,
                'toolChoice' => 'required',
                'model' => AnthropicProvider::MODEL_CLAUDE_FABLE_5_1,
            ]))->toThrow(ProviderException::class);
        });

        it('still allows auto and none, which the model accepts', function () {
            foreach (['auto', 'none'] as $choice) {
                $this->provider->chat([Message::user('hi')], [
                    'tools' => $this->tools,
                    'toolChoice' => $choice,
                    'model' => AnthropicProvider::MODEL_CLAUDE_FABLE_5_1,
                ]);

                expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => $choice]);
            }
        });
    });

    describe('on the 5.5 generation and on unknown IDs', function () {
        beforeEach(function () {
            $this->provider = new TestableAnthropicToolChoiceProvider('test-api-key');
            $this->tools = [
                ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object']],
            ];
        });

        it('throws for a forced choice on Opus 5.5 and Sonnet 5.5', function () {
            foreach ([AnthropicProvider::MODEL_CLAUDE_OPUS_5_5, AnthropicProvider::MODEL_CLAUDE_SONNET_5_5] as $model) {
                expect(fn () => $this->provider->chat([Message::user('hi')], [
                    'tools' => $this->tools, 'toolChoice' => 'required', 'model' => $model,
                ]))->toThrow(ProviderException::class, 'forced tool choice');
            }
        });

        it('throws for a forced choice on a model it has not heard of, since the newest generations refuse', function () {
            expect(fn () => $this->provider->chat([Message::user('hi')], [
                'tools' => $this->tools, 'toolChoice' => ['name' => 'get_weather'], 'model' => 'claude-opus-6',
            ]))->toThrow(ProviderException::class);
        });

        it('still forwards a forced choice on Sonnet 5, which accepts it', function () {
            $this->provider->chat([Message::user('hi')], [
                'tools' => $this->tools, 'toolChoice' => 'required', 'model' => AnthropicProvider::MODEL_CLAUDE_SONNET_5,
            ]);

            expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'any']);
        });
    });
});
