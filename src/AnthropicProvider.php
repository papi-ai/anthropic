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
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use RuntimeException;

class AnthropicProvider implements ProviderInterface
{
    private const API_URL = 'https://api.anthropic.com/v1/messages';
    private const API_VERSION = '2023-06-01';

    protected ?int $lastRetryAfter = null;

    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = 'claude-sonnet-4-20250514',
        private readonly int $defaultMaxTokens = 4096,
    ) {
    }

    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromAnthropic($response, $messages);
    }

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

    public function supportsTool(): bool
    {
        return true;
    }

    public function supportsVision(): bool
    {
        return true;
    }

    public function supportsStructuredOutput(): bool
    {
        return false; // Anthropic doesn't have native JSON mode yet
    }

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
            $payload['tools'] = $options['tools'];
        }

        return $payload;
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
     * Make an API request.
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

        curl_close($ch);

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
     * @throws AuthenticationException
     * @throws RateLimitException
     * @throws ProviderException
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
     * Make a streaming API request.
     *
     * @return Generator<array>
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
        curl_close($ch);

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
