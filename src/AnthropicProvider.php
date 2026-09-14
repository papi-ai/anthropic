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

namespace PapiAI\Anthropic;

use Generator;
use PapiAI\Core\Contracts\NamedToolSelectableInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Effort;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Exception\UnknownEffortException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\ToolChoice;
use RuntimeException;

/**
 * Anthropic Messages API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with Anthropic's Messages API,
 * handling format conversion in both directions. Supports chat completions, streaming,
 * tool calling, vision (multimodal), structured output, and prompt caching (cache_control).
 *
 * Authentication is via API key passed in the x-api-key header. All HTTP is done with ext-curl
 * directly, with no HTTP abstraction layer. Respects retry-after headers for rate limiting.
 *
 * @see https://docs.anthropic.com/en/docs
 */
class AnthropicProvider implements ProviderInterface, NamedToolSelectableInterface
{
    private const API_URL = 'https://api.anthropic.com/v1/messages';
    private const API_VERSION = '2023-06-01';

    // Current generation. These IDs are dateless pinned snapshots, not moving aliases.
    public const MODEL_CLAUDE_FABLE_5_1 = 'claude-fable-5-1';
    public const MODEL_CLAUDE_OPUS_5 = 'claude-opus-5';
    public const MODEL_CLAUDE_SONNET_5 = 'claude-sonnet-5';
    public const MODEL_CLAUDE_FABLE_5 = 'claude-fable-5';
    public const MODEL_CLAUDE_HAIKU_4_5 = 'claude-haiku-4-5';

    // Previous generations, still active.
    public const MODEL_CLAUDE_OPUS_4_8 = 'claude-opus-4-8';
    public const MODEL_CLAUDE_OPUS_4_7 = 'claude-opus-4-7';
    public const MODEL_CLAUDE_OPUS_4_6 = 'claude-opus-4-6';
    public const MODEL_CLAUDE_SONNET_4_6 = 'claude-sonnet-4-6';

    /**
     * The two levels Anthropic spells differently from the neutral scale.
     */
    private const NATIVE_EFFORT = [
        'extra-high' => 'xhigh',
        'maximum' => 'max',
    ];

    protected ?int $lastRetryAfter = null;

    /**
     * @param string      $apiKey       Anthropic API key for authentication
     * @param string      $defaultModel Default model identifier
     * @param int         $defaultMaxTokens Default maximum tokens for responses
     * @param Effort|null $defaultEffort Extended-thinking effort when none is given per call
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_CLAUDE_SONNET_5,
        private readonly int $defaultMaxTokens = 4096,
        private readonly ?Effort $defaultEffort = null,
    ) {
    }

    /**
     * Send a chat completion request to the Anthropic Messages API.
     *
     * @param Message[] $messages Conversation messages
     * @param array     $options  Options including model, maxTokens, temperature, stopSequences, tools, and cache
     *
     * @return Response The parsed API response
     *
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limited by the API
     * @throws ProviderException       When the API returns an error
     */
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return $this->mapResponse($response, $messages);
    }

    /**
     * Map an Anthropic Messages API payload to a neutral Response.
     *
     * @param array          $payload  The decoded Anthropic API response
     * @param array<Message> $messages Conversation history to attach
     *
     * @return Response The neutral response
     */
    private function mapResponse(array $payload, array $messages): Response
    {
        $text = '';
        $toolCalls = [];

        foreach ($payload['content'] ?? [] as $block) {
            if (($block['type'] ?? null) === 'text') {
                $text .= $block['text'];
            } elseif (($block['type'] ?? null) === 'tool_use') {
                $toolCalls[] = new ToolCall($block['id'], $block['name'], $block['input'] ?? []);
            }
        }

        return new Response(
            text: $text,
            toolCalls: $toolCalls,
            messages: $messages,
            usage: $payload['usage'] ?? [],
            stopReason: $payload['stop_reason'] ?? null,
        );
    }

    /**
     * Stream a chat completion response from the Anthropic Messages API.
     *
     * Yields StreamChunk objects as content blocks arrive via server-sent events.
     *
     * @param Message[] $messages Conversation messages
     * @param array     $options  Options including model, maxTokens, temperature, stopSequences, tools, and cache
     *
     * @return iterable<StreamChunk> Stream of response chunks
     *
     * @throws RuntimeException When the HTTP request fails
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            if ($event['type'] === 'content_block_delta') {
                $delta = $event['delta'] ?? [];
                if (isset($delta['text'])) {
                    yield new StreamChunk($delta['text']);
                }
            } elseif ($event['type'] === 'message_stop') {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    /**
     * Whether this provider supports tool calling.
     *
     * @return bool Always true; Anthropic supports native tool use
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports vision (multimodal image input).
     *
     * @return bool Always true; Anthropic supports base64 and URL image inputs
     */
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports structured JSON output.
     *
     * @return bool Always false; Anthropic does not have a native JSON mode
     */
    public function supportsStructuredOutput(): bool
    {
        return false; // Anthropic doesn't have native JSON mode yet
    }

    /**
     * Get the provider identifier.
     *
     * @return string The provider name "anthropic"
     */
    public function getName(): string
    {
        return 'anthropic';
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $systemMessage = null;
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                if ($message->isSystem()) {
                    $systemMessage = $message->getText();
                    continue;
                }

                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'model' => $options['model'] ?? $this->defaultModel,
            'max_tokens' => $options['maxTokens'] ?? $this->defaultMaxTokens,
            'messages' => $apiMessages,
        ];

        if ($systemMessage !== null) {
            if (isset($options['cache']) && $options['cache'] === true) {
                $payload['system'] = [
                    [
                        'type' => 'text',
                        'text' => $systemMessage,
                        'cache_control' => ['type' => 'ephemeral'],
                    ],
                ];
            } else {
                $payload['system'] = $systemMessage;
            }
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop_sequences'] = $options['stopSequences'];
        }

        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        // Forced tool choice. Validation lives in core and throws before any HTTP call.
        if (isset($options['toolChoice'])) {
            $choice = ToolChoice::fromOption($options['toolChoice'], $options['tools'] ?? []);

            if (!empty($options['tools'])) {
                $this->assertCanForce($choice, (string) $payload['model']);

                $payload['tool_choice'] = $choice->toolName !== null
                    ? ['type' => 'tool', 'name' => $choice->toolName]
                    : match ($choice->mode) {
                        ToolChoice::NONE => ['type' => 'none'],
                        ToolChoice::REQUIRED => ['type' => 'any'],
                        default => ['type' => 'auto'],
                    };
            }
        }

        $effort = $this->effortFor($options);

        if ($effort !== null) {
            $payload = $this->withThinking($payload, $effort, (string) $payload['model']);
        }

        return $payload;
    }

    /**
     * Refuse a forced tool choice on the models that return 400 for one.
     *
     * Fable 5.1 accepts "auto" and "none" only. Throwing here, before the round trip, says why;
     * the API's own error does not.
     *
     * @throws ProviderException When the model cannot honour "any" or a named tool
     */
    private function assertCanForce(ToolChoice $choice, string $model): void
    {
        if ($choice->isAuto() || $choice->mode === ToolChoice::NONE || !$this->refusesForcedTools($model)) {
            return;
        }

        throw new ProviderException(
            sprintf(
                'Model "%s" does not accept a forced tool choice: the API returns 400 for "required" and for a named tool. Use "auto" with an instruction, or structured output.',
                $model,
            ),
            $this->getName(),
        );
    }

    /**
     * Apply a level of effort in whichever shape this model's generation accepts.
     *
     * Three generations, three shapes. Haiku 4.5 predates adaptive thinking and wants a token
     * budget. Everything from 4.6 on wants adaptive thinking plus an effort level, and rejects
     * a budget with a 400. Fable cannot stop thinking at all, so it never gets a thinking block.
     *
     * @param array<string, mixed> $payload The request so far
     *
     * @return array<string, mixed> The request with thinking applied
     *
     * @throws ProviderException When a budget model's ceiling is too small to think and answer
     */
    private function withThinking(array $payload, Effort $effort, string $model): array
    {
        if ($this->thinksAlways($model)) {
            // Cannot be switched off, so "none" narrows to the shallowest level on offer.
            $payload['output_config'] = ['effort' => $this->nativeLevel($effort, $model)];

            return $payload;
        }

        if ($this->takesBudget($model)) {
            return $this->withBudget($payload, $effort);
        }

        if (!$effort->thinks()) {
            // Sonnet 5 and Opus 5 think unless told not to. The 4.x models are off when the
            // field is omitted, so saying "disabled" there would only be noise.
            if ($this->thinksByDefault($model)) {
                $payload['thinking'] = ['type' => 'disabled'];
            }

            return $payload;
        }

        $payload['thinking'] = ['type' => 'adaptive'];
        $payload['output_config'] = ['effort' => $this->nativeLevel($effort, $model)];

        return $payload;
    }

    /**
     * The pre-4.6 shape: a token budget carved out of max_tokens.
     *
     * @param array<string, mixed> $payload The request so far
     *
     * @return array<string, mixed> The request with a thinking budget, or untouched for "none"
     *
     * @throws ProviderException When max_tokens cannot hold both the budget and an answer
     */
    private function withBudget(array $payload, Effort $effort): array
    {
        if (!$effort->thinks()) {
            return $payload;
        }

        $maxTokens = (int) $payload['max_tokens'];

        if (!$effort->fitsWithin($maxTokens)) {
            throw new ProviderException(
                sprintf(
                    'Extended thinking needs at least %d tokens for thinking plus room to answer, but maxTokens is %d. Raise maxTokens or ask for "none".',
                    Effort::MINIMUM_BUDGET,
                    $maxTokens,
                ),
                $this->getName(),
            );
        }

        $payload['thinking'] = ['type' => 'enabled', 'budget_tokens' => $effort->budgetWithin($maxTokens)];

        return $payload;
    }

    /**
     * The effort level this model accepts that is nearest to the one asked for, in Anthropic's spelling.
     */
    private function nativeLevel(Effort $effort, string $model): string
    {
        $offered = $this->lacksExtraHigh($model)
            ? [Effort::Low, Effort::Medium, Effort::High, Effort::Maximum]
            : [Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum];

        $level = $effort->nearestOf($offered)->value;

        return self::NATIVE_EFFORT[$level] ?? $level;
    }

    /**
     * Fable and Mythos: thinking is always on and a "disabled" block is a 400.
     */
    private function thinksAlways(string $model): bool
    {
        return preg_match('/claude-(fable|mythos)-/i', $model) === 1;
    }

    /**
     * Pre-4.6 models, which take a token budget and reject an effort level.
     *
     * Anything unrecognised is assumed to be newer, not older: a model we have not heard of is
     * far more likely to be next month's than last year's.
     */
    private function takesBudget(string $model): bool
    {
        return preg_match('/claude-(haiku-4-5|3-|opus-4-[15]|sonnet-4-5|(opus|sonnet)-4-2025)/i', $model) === 1;
    }

    /**
     * Sonnet 5 and Opus 5 run adaptive thinking when the field is omitted.
     */
    private function thinksByDefault(string $model): bool
    {
        return preg_match('/claude-(opus|sonnet)-5(?![0-9.])/i', $model) === 1;
    }

    /**
     * The 4.6 pair predates the xhigh level.
     */
    private function lacksExtraHigh(string $model): bool
    {
        return preg_match('/claude-(opus|sonnet)-4-6(?![0-9])/i', $model) === 1;
    }

    /**
     * Fable 5.1 and Mythos 5.1 return 400 for "any" and for a named tool.
     */
    private function refusesForcedTools(string $model): bool
    {
        return preg_match('/claude-(fable|mythos)-5-1(?![0-9])/i', $model) === 1;
    }

    /**
     * The effort this request asks for: the per-call option, else the provider default.
     *
     * @param array<string, mixed> $options The caller's request options
     *
     * @throws UnknownEffortException When the level is not one core defines
     */
    private function effortFor(array $options): ?Effort
    {
        if (!isset($options['effort'])) {
            return $this->defaultEffort;
        }

        $level = (string) $options['effort'];

        return Effort::tryFrom($level) ?? throw new UnknownEffortException($level);
    }

    /**
     * Convert neutral tool definitions to Anthropic's tool format.
     *
     * Accepts the neutral shape the Agent emits (name, description, parameters) and also a pre-built
     * Anthropic shape (input_schema) for backward compatibility.
     *
     * @param array<array<string, mixed>> $tools Neutral tool definitions
     *
     * @return array<array{name: string, description: string, input_schema: array}>
     */
    private function convertTools(array $tools): array
    {
        return array_map(
            fn (array $tool): array => [
                'name' => $tool['name'],
                'description' => $tool['description'] ?? '',
                'input_schema' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
            ],
            $tools,
        );
    }

    /**
     * Convert a Message to Anthropic API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            // Tool result message
            $apiMessage['content'] = [
                [
                    'type' => 'tool_result',
                    'tool_use_id' => $message->toolCallId,
                    'content' => $message->content,
                ],
            ];
        } elseif ($message->hasToolCalls()) {
            // Assistant message with tool calls
            $content = [];
            if ($message->getText() !== '') {
                $content[] = ['type' => 'text', 'text' => $message->getText()];
            }
            foreach ($message->toolCalls as $toolCall) {
                $content[] = [
                    'type' => 'tool_use',
                    'id' => $toolCall->id,
                    'name' => $toolCall->name,
                    'input' => $toolCall->arguments,
                ];
            }
            $apiMessage['content'] = $content;
        } elseif (is_array($message->content)) {
            // Multimodal content
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            // Simple text content
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to Anthropic format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $result = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $result[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    // Anthropic doesn't support URL images directly, need to fetch
                    $result[] = [
                        'type' => 'image',
                        'source' => [
                            'type' => 'url',
                            'url' => $source['url'],
                        ],
                    ];
                } else {
                    $result[] = [
                        'type' => 'image',
                        'source' => [
                            'type' => 'base64',
                            'media_type' => $source['media_type'],
                            'data' => $source['data'],
                        ],
                    ];
                }
            }
        }

        return $result;
    }

    /**
     * Convert Role to Anthropic role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::User, Role::Tool => 'user',
            Role::Assistant => 'assistant',
            Role::System => 'user', // System is handled separately
        };
    }

    /**
     * Send a synchronous POST request to the Anthropic Messages API.
     *
     * Captures response headers (including retry-after) and decodes the JSON response.
     *
     * @param array $payload The JSON-encodable request body
     *
     * @return array The decoded API response
     *
     * @throws RuntimeException        When the cURL request fails
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limited (HTTP 429)
     * @throws ProviderException       When the API returns any other error
     */
    protected function request(array $payload): array
    {
        $ch = curl_init(self::API_URL);

        $responseHeaders = [];
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'x-api-key: ' . $this->apiKey,
                'anthropic-version: ' . self::API_VERSION,
            ],
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$responseHeaders) {
                $parts = explode(':', $header, 2);
                if (count($parts) === 2) {
                    $responseHeaders[strtolower(trim($parts[0]))] = trim($parts[1]);
                }

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        $this->lastRetryAfter = isset($responseHeaders['retry-after'])
            ? (int) $responseHeaders['retry-after']
            : null;

        if ($error !== '') {
            throw new RuntimeException("Anthropic API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data, $response);
        }

        return $data;
    }

    /**
     * Throw the appropriate exception based on HTTP status code.
     *
     * Maps 401 to AuthenticationException, 429 to RateLimitException (with retry-after),
     * and all other 4xx/5xx codes to ProviderException.
     *
     * @param int         $httpCode    The HTTP response status code
     * @param array|null  $data        The decoded response body, if available
     * @param string      $rawResponse The raw response string for fallback error messages
     *
     * @throws AuthenticationException When HTTP 401 (invalid API key)
     * @throws RateLimitException      When HTTP 429 (rate limited)
     * @throws ProviderException       For all other error status codes
     */
    protected function throwForStatusCode(int $httpCode, ?array $data, string $rawResponse): never
    {
        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        if ($httpCode === 401) {
            throw new AuthenticationException(
                $this->getName(),
                $httpCode,
                $data,
            );
        }

        if ($httpCode === 429) {
            throw new RateLimitException(
                $this->getName(),
                $this->lastRetryAfter,
                $httpCode,
                $data,
            );
        }

        throw new ProviderException(
            "Anthropic API error ({$httpCode}): {$errorMessage}",
            $this->getName(),
            $httpCode,
            $data,
        );
    }

    /**
     * Send a streaming POST request to the Anthropic Messages API.
     *
     * Buffers the full SSE response then parses and yields individual events.
     *
     * @param array $payload The JSON-encodable request body (must include stream: true)
     *
     * @return Generator<int, array> Parsed SSE event objects
     */
    protected function streamRequest(array $payload): Generator
    {
        $ch = curl_init(self::API_URL);

        $buffer = '';

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'x-api-key: ' . $this->apiKey,
                'anthropic-version: ' . self::API_VERSION,
                'Accept: text/event-stream',
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);
        curl_exec($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }
}
