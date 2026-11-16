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
use PapiAI\Anthropic\Compaction;
use PapiAI\Anthropic\ThinkingReplay;
use PapiAI\Anthropic\ToolResultClearing;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Message;
use PapiAI\Core\ToolCall;

/**
 * Captures the request so what goes back to the API can be asserted without HTTP.
 */
class TestableAnthropicReplayProvider extends AnthropicProvider
{
    public array $lastPayload = [];
    public array $fakeResponse = ['content' => [['type' => 'text', 'text' => 'ok']], 'stop_reason' => 'end_turn'];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    public function headersSent(): array
    {
        return $this->headersFor($this->lastPayload);
    }
}

dataset('a tool turn with thinking', [[[
    ['type' => 'thinking', 'thinking' => '', 'signature' => 'sig-1'],
    ['type' => 'text', 'text' => 'Let me look.'],
    ['type' => 'tool_use', 'id' => 'tu_1', 'name' => 'status', 'input' => []],
]]]);

describe('AnthropicProvider replay', function () {
    beforeEach(function () {
        $this->provider = fn (ThinkingReplay $replay = ThinkingReplay::Off, array $contextEdits = []) => new TestableAnthropicReplayProvider(
            'test-api-key',
            thinkingReplay: $replay,
            contextEdits: $contextEdits,
        );

        $this->toolTurn = fn (array $content, string $model = 'claude-sonnet-5-5', array $options = []): array => [
            [
                Message::system('You pair on code.'),
                Message::user('What is the status?'),
                Message::assistant('Let me look.', [new ToolCall('tu_1', 'status', [])], ['anthropic' => $content]),
                Message::toolResult('tu_1', 'green'),
            ],
            $options + ['model' => $model],
        ];
    });

    describe('what a response leaves behind', function () {
        it('leaves its content on the response when it holds thinking, for the next request', function (array $content) {
            $provider = ($this->provider)();
            $provider->fakeResponse = ['content' => $content, 'stop_reason' => 'tool_use'];

            $response = $provider->chat([Message::user('What is the status?')]);

            expect($response->replay)->toBe(['anthropic' => $content]);
            expect($response->toMessage()->replayFor('anthropic'))->toBe($content);
        })->with('a tool turn with thinking');

        it('leaves nothing when there is no thinking or compaction to send back', function () {
            $provider = ($this->provider)();

            expect($provider->chat([Message::user('Hi')])->replay)->toBe([]);
        });

        it('counts what a compaction spent along with the answer', function () {
            $provider = ($this->provider)();
            $provider->fakeResponse = [
                'content' => [['type' => 'compaction', 'content' => 'Summary'], ['type' => 'text', 'text' => 'ok']],
                'stop_reason' => 'end_turn',
                'usage' => [
                    'input_tokens' => 2000,
                    'output_tokens' => 100,
                    'iterations' => [
                        ['type' => 'compaction', 'input_tokens' => 180000, 'output_tokens' => 3000],
                        ['type' => 'message', 'input_tokens' => 2000, 'output_tokens' => 100],
                    ],
                ],
            ];

            $response = $provider->chat([Message::user('Hi')]);

            expect($response->getInputTokens())->toBe(182000);
            expect($response->getOutputTokens())->toBe(3100);
        });
    });

    describe('thinking', function () {
        it('is not sent back unless asked for', function (array $content) {
            $provider = ($this->provider)();

            $provider->chat(...($this->toolTurn)($content));

            $assistant = $provider->lastPayload['messages'][1];
            expect(array_column($assistant['content'], 'type'))->toBe(['text', 'tool_use']);
            expect(json_encode($assistant))->toContain('"input":{}');
            expect($provider->lastPayload)->not->toHaveKey('thinking');
        })->with('a tool turn with thinking');

        it('goes back exactly as it came, in order, with a failing binding when asked for', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $provider->chat(...($this->toolTurn)($content, options: ['effort' => 'high']));

            $assistant = $provider->lastPayload['messages'][1];
            expect(array_column($assistant['content'], 'type'))->toBe(['thinking', 'text', 'tool_use']);
            expect($assistant['content'][0])->toBe($content[0]);
            expect(json_encode($assistant))->toContain('"input":{}');
            expect($provider->lastPayload['thinking'])->toBe([
                'type' => 'adaptive',
                'block_binding' => ['prefix_mismatch_behavior' => 'error'],
            ]);
            expect($provider->lastPayload['output_config'])->toBe(['effort' => 'high']);
            expect($provider->headersSent())->toContain('anthropic-beta: thinking-binding-controls-2026-08-01');
        })->with('a tool turn with thinking');

        it('asks the API to drop what a history edit invalidated', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::DropOnEdit);

            $provider->chat(...($this->toolTurn)($content));

            expect($provider->lastPayload['thinking']['block_binding'])->toBe(['prefix_mismatch_behavior' => 'drop_block']);
        })->with('a tool turn with thinking');

        it('binds on a model that thinks by default, even when no effort is asked for', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $provider->chat(...($this->toolTurn)($content));

            expect($provider->lastPayload['thinking']['type'])->toBe('adaptive');
            expect($provider->lastPayload)->not->toHaveKey('output_config');
        })->with('a tool turn with thinking');

        it('binds on a model whose thinking is always on', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $provider->chat(...($this->toolTurn)($content, 'claude-fable-5-2', ['effort' => 'high']));

            expect($provider->lastPayload['thinking']['type'])->toBe('adaptive');
            expect($provider->lastPayload['output_config'])->toBe(['effort' => 'high']);
        })->with('a tool turn with thinking');

        it('binds on a model it has not heard of, assuming the newest rules', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $provider->chat(...($this->toolTurn)($content, 'claude-future-6'));

            expect($provider->lastPayload['thinking']['type'])->toBe('adaptive');
        })->with('a tool turn with thinking');

        it('leaves thinking out rather than risk a 400, when dropping is asked for but cannot be bound', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::DropOnEdit);

            $provider->chat(...($this->toolTurn)($content, options: ['effort' => 'none']));

            expect($provider->lastPayload['thinking'])->toBe(['type' => 'between_tools']);
            expect(array_column($provider->lastPayload['messages'][1]['content'], 'type'))->toBe(['text', 'tool_use']);
            expect($provider->headersSent())->not->toContain('anthropic-beta: thinking-binding-controls-2026-08-01');
        })->with('a tool turn with thinking');

        it('sends thinking back unbound where no binding can be sent, when failing is asked for', function (array $content) {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $provider->chat(...($this->toolTurn)($content, 'claude-opus-4-8'));

            expect($provider->lastPayload)->not->toHaveKey('thinking');
            expect(array_column($provider->lastPayload['messages'][1]['content'], 'type'))->toBe(['thinking', 'text', 'tool_use']);
        })->with('a tool turn with thinking');

        it('never sends another provider\'s state', function () {
            $provider = ($this->provider)(ThinkingReplay::FailOnEdit);

            $withOpenAI = [
                Message::user('What is the status?'),
                Message::assistant('Let me look.', [new ToolCall('tu_1', 'status', [])], ['openai' => [['type' => 'reasoning', 'id' => 'rs_1']]]),
                Message::toolResult('tu_1', 'green'),
            ];
            $provider->chat($withOpenAI, ['model' => 'claude-sonnet-5-5']);

            expect(json_encode($provider->lastPayload['messages']))->not->toContain('reasoning');
        });
    });

    describe('a system message after the conversation has started', function () {
        it('stays in place on a model that takes it, leaving the opening prompt alone', function () {
            $provider = ($this->provider)();

            $provider->chat([
                Message::system('You are the navigator.'),
                Message::user('Start.'),
                Message::assistant('Started.'),
                Message::user('Swap roles.'),
                Message::system('You are now the driver.'),
            ], ['model' => 'claude-sonnet-5-5']);

            expect($provider->lastPayload['system'])->toBe('You are the navigator.');
            expect(end($provider->lastPayload['messages']))->toBe(['role' => 'system', 'content' => 'You are now the driver.']);
        });

        it('stays in place on a model it has not heard of', function () {
            $provider = ($this->provider)();

            $provider->chat([Message::user('Start.'), Message::system('Be brief.')], ['model' => 'claude-future-6']);

            expect($provider->lastPayload['messages'][1]['role'])->toBe('system');
        });

        it('replaces the system prompt on a model that cannot take it', function () {
            $provider = ($this->provider)();

            $provider->chat([
                Message::system('You are the navigator.'),
                Message::user('Start.'),
                Message::system('You are now the driver.'),
            ], ['model' => 'claude-sonnet-5']);

            expect($provider->lastPayload['system'])->toBe('You are now the driver.');
            expect(array_column($provider->lastPayload['messages'], 'role'))->toBe(['user']);
        });

        it('may come just before an assistant turn', function () {
            $provider = ($this->provider)();

            $provider->chat([
                Message::user('Start.'),
                Message::system('Be brief.'),
                Message::assistant('Started.'),
                Message::user('Go on.'),
            ], ['model' => 'claude-sonnet-5-5']);

            expect(array_column($provider->lastPayload['messages'], 'role'))->toBe(['user', 'system', 'assistant', 'user']);
        });

        it('is refused before any request where the API would reject it', function () {
            $provider = ($this->provider)();

            expect(fn () => $provider->chat([
                Message::user('Start.'),
                Message::assistant('Started.'),
                Message::system('You are now the driver.'),
                Message::user('Go on.'),
            ], ['model' => 'claude-sonnet-5-5']))->toThrow(
                ProviderException::class,
                'The system message at position 2 must follow a user message and be the last message or come just before an assistant one. Append it after the next user message instead.',
            );
            expect($provider->lastPayload)->toBe([]);
        });
    });

    describe('context management on the server', function () {
        it('sends no beta header for a plain request', function () {
            $provider = ($this->provider)();

            $provider->chat([Message::user('Hi')]);

            expect(implode("\n", $provider->headersSent()))->not->toContain('anthropic-beta');
        });

        it('compacts when the conversation grows past the trigger', function () {
            $provider = ($this->provider)(contextEdits: [new Compaction(80000, 'Keep every file path.')]);

            $provider->chat([Message::user('Hi')]);

            expect($provider->lastPayload['context_management'])->toBe(['edits' => [[
                'type' => 'compact_20260112',
                'trigger' => ['type' => 'input_tokens', 'value' => 80000],
                'instructions' => 'Keep every file path.',
            ]]]);
            expect($provider->headersSent())->toContain('anthropic-beta: compact-2026-01-12');
        });

        it('compacts at the API\'s own default trigger unless told otherwise', function () {
            expect((new Compaction())->edit())->toBe([
                'type' => 'compact_20260112',
                'trigger' => ['type' => 'input_tokens', 'value' => 150000],
            ]);
        });

        it('refuses a trigger below the API\'s minimum', function () {
            expect(fn () => new Compaction(10000))
                ->toThrow(InvalidArgumentException::class, 'Compaction cannot trigger below 50000 input tokens; 10000 given.');
        });

        it('clears old tool results', function () {
            expect((new ToolResultClearing(30000, 2, ['web_search']))->edit())->toBe([
                'type' => 'clear_tool_uses_20250919',
                'trigger' => ['type' => 'input_tokens', 'value' => 30000],
                'keep' => ['type' => 'tool_uses', 'value' => 2],
                'exclude_tools' => ['web_search'],
            ]);
        });

        it('sends each edit with the beta it needs', function () {
            $provider = ($this->provider)(contextEdits: [new ToolResultClearing(), new Compaction()]);

            $provider->chat([Message::user('Hi')]);

            expect(array_column($provider->lastPayload['context_management']['edits'], 'type'))->toBe(['clear_tool_uses_20250919', 'compact_20260112']);
            expect($provider->headersSent())->toContain('anthropic-beta: context-management-2025-06-27,compact-2026-01-12');
        });

        it('sends the compaction block back, whatever happens to thinking', function () {
            $provider = ($this->provider)();
            $compacted = [['type' => 'compaction', 'content' => 'Summary'], ['type' => 'text', 'text' => 'Done.']];

            $provider->chat([
                Message::user('Start.'),
                Message::assistant('Done.', replay: ['anthropic' => $compacted]),
                Message::user('Next.'),
            ]);

            expect($provider->lastPayload['messages'][1]['content'])->toBe($compacted);
            expect($provider->headersSent())->toContain('anthropic-beta: compact-2026-01-12');
        });
    });
});
