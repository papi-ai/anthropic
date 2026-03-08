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
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;

/**
 * Test subclass that stubs HTTP methods for unit testing.
 */
class TestableAnthropicProvider extends AnthropicProvider
{
    public array $lastPayload = [];
    public array $fakeResponse = [];
    public array $fakeStreamEvents = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;

        foreach ($this->fakeStreamEvents as $event) {
            yield $event;
        }
    }

    public function callThrowForStatusCode(int $httpCode, ?array $data, string $rawResponse = ''): never
    {
        $this->throwForStatusCode($httpCode, $data, $rawResponse);
    }

    public function setLastRetryAfter(?int $value): void
    {
        $this->lastRetryAfter = $value;
    }
}

describe('AnthropicProvider', function () {
    beforeEach(function () {
        $this->provider = new TestableAnthropicProvider('test-api-key');
    });

    describe('construction', function () {
        it('implements ProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('returns anthropic as name', function () {
            expect($this->provider->getName())->toBe('anthropic');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            expect($this->provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            expect($this->provider->supportsVision())->toBeTrue();
        });

        it('does not support native structured output', function () {
            expect($this->provider->supportsStructuredOutput())->toBeFalse();
        });
    });

    describe('chat', function () {
        it('sends messages and returns a Response', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'Hello back!']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response)->toBeInstanceOf(Response::class);
            expect($response->text)->toBe('Hello back!');
        });

        it('includes system message in payload', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'OK']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $this->provider->chat([
                Message::system('Be helpful'),
                Message::user('Hello'),
            ]);

            expect($this->provider->lastPayload['system'])->toBe('Be helpful');
            expect($this->provider->lastPayload['messages'])->toHaveCount(1);
            expect($this->provider->lastPayload['messages'][0]['role'])->toBe('user');
        });

        it('uses default model and max tokens', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'OK']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')]);

            expect($this->provider->lastPayload['model'])->toBe('claude-sonnet-4-20250514');
            expect($this->provider->lastPayload['max_tokens'])->toBe(4096);
        });

        it('overrides model and options from parameters', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'OK']],
                'model' => 'claude-3-opus-20240229',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')], [
                'model' => 'claude-3-opus-20240229',
                'maxTokens' => 8192,
                'temperature' => 0.5,
                'stopSequences' => ['END'],
            ]);

            expect($this->provider->lastPayload['model'])->toBe('claude-3-opus-20240229');
            expect($this->provider->lastPayload['max_tokens'])->toBe(8192);
            expect($this->provider->lastPayload['temperature'])->toBe(0.5);
            expect($this->provider->lastPayload['stop_sequences'])->toBe(['END']);
        });

        it('includes tools in payload', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'OK']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $tools = [
                [
                    'name' => 'get_weather',
                    'description' => 'Get weather',
                    'input_schema' => ['type' => 'object', 'properties' => []],
                ],
            ];

            $this->provider->chat([Message::user('Hello')], ['tools' => $tools]);

            expect($this->provider->lastPayload['tools'])->toBe($tools);
        });

        it('converts tool result messages', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'The weather is sunny']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('What is the weather?'),
                Message::assistant('Let me check', [
                    new ToolCall('tc_1', 'get_weather', ['city' => 'London']),
                ]),
                Message::toolResult('tc_1', ['temp' => 20]),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            expect($messages)->toHaveCount(3);

            // Tool result message
            $toolMsg = $messages[2];
            expect($toolMsg['role'])->toBe('user');
            expect($toolMsg['content'][0]['type'])->toBe('tool_result');
            expect($toolMsg['content'][0]['tool_use_id'])->toBe('tc_1');
        });

        it('converts assistant messages with tool calls', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'Done']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('Hello'),
                Message::assistant('Let me help', [
                    new ToolCall('tc_1', 'search', ['q' => 'test']),
                ]),
                Message::toolResult('tc_1', 'result'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $assistantMsg = $messages[1];
            expect($assistantMsg['role'])->toBe('assistant');
            expect($assistantMsg['content'][0]['type'])->toBe('text');
            expect($assistantMsg['content'][1]['type'])->toBe('tool_use');
            expect($assistantMsg['content'][1]['name'])->toBe('search');
        });

        it('converts multimodal messages with url images', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'I see a cat']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 100, 'output_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'https://example.com/cat.jpg'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content)->toBeArray();
            expect($content[0]['type'])->toBe('text');
            expect($content[1]['type'])->toBe('image');
            expect($content[1]['source']['type'])->toBe('url');
        });

        it('converts multimodal messages with base64 images', function () {
            $this->provider->fakeResponse = [
                'content' => [['type' => 'text', 'text' => 'I see a cat']],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 100, 'output_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'base64data', 'image/png'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content[1]['source']['type'])->toBe('base64');
            expect($content[1]['source']['media_type'])->toBe('image/png');
        });

        it('handles response with tool calls', function () {
            $this->provider->fakeResponse = [
                'content' => [
                    ['type' => 'text', 'text' => 'Let me check'],
                    [
                        'type' => 'tool_use',
                        'id' => 'toolu_123',
                        'name' => 'get_weather',
                        'input' => ['city' => 'London'],
                    ],
                ],
                'model' => 'claude-sonnet-4-20250514',
                'role' => 'assistant',
                'stop_reason' => 'tool_use',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 20],
            ];

            $response = $this->provider->chat([Message::user('Weather?')]);

            expect($response->hasToolCalls())->toBeTrue();
            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->name)->toBe('get_weather');
        });
    });

    describe('stream', function () {
        it('yields StreamChunk for text deltas', function () {
            $this->provider->fakeStreamEvents = [
                ['type' => 'content_block_delta', 'delta' => ['text' => 'Hello']],
                ['type' => 'content_block_delta', 'delta' => ['text' => ' world']],
                ['type' => 'message_stop'],
            ];

            $chunks = [];
            foreach ($this->provider->stream([Message::user('Hi')]) as $chunk) {
                $chunks[] = $chunk;
            }

            expect($chunks)->toHaveCount(3);
            expect($chunks[0])->toBeInstanceOf(StreamChunk::class);
            expect($chunks[0]->text)->toBe('Hello');
            expect($chunks[1]->text)->toBe(' world');
            expect($chunks[2]->isComplete)->toBeTrue();
        });

        it('sets stream flag in payload', function () {
            $this->provider->fakeStreamEvents = [
                ['type' => 'message_stop'],
            ];

            iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastPayload['stream'])->toBeTrue();
        });

        it('ignores non-text delta events', function () {
            $this->provider->fakeStreamEvents = [
                ['type' => 'message_start', 'message' => []],
                ['type' => 'content_block_start'],
                ['type' => 'content_block_delta', 'delta' => ['text' => 'Hi']],
                ['type' => 'content_block_stop'],
                ['type' => 'message_stop'],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($chunks)->toHaveCount(2); // text + complete
        });
    });

    describe('error mapping', function () {
        it('throws AuthenticationException on 401', function () {
            $this->provider->callThrowForStatusCode(401, ['error' => ['message' => 'Invalid key']], '');
        })->throws(AuthenticationException::class);

        it('throws RateLimitException on 429', function () {
            $this->provider->callThrowForStatusCode(429, ['error' => ['message' => 'Too many requests']], '');
        })->throws(RateLimitException::class);

        it('throws RateLimitException on 429 with retry-after', function () {
            $this->provider->setLastRetryAfter(30);
            $this->provider->callThrowForStatusCode(429, ['error' => ['message' => 'Too many requests']], '');
        })->throws(RateLimitException::class);

        it('throws ProviderException on 500', function () {
            $this->provider->callThrowForStatusCode(500, ['error' => ['message' => 'Server error']], '');
        })->throws(ProviderException::class);

        it('throws ProviderException with unknown error for null data', function () {
            $this->provider->callThrowForStatusCode(500, null, '');
        })->throws(ProviderException::class);
    });
});
