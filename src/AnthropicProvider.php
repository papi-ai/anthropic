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
use PapiAI\Core\StopReason;
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
    public const MODEL_CLAUDE_FABLE_5_2 = ClaudeModel::Fable52->value;
    public const MODEL_CLAUDE_FABLE_5_1 = ClaudeModel::Fable51->value;
    public const MODEL_CLAUDE_MYTHOS_5_2 = ClaudeModel::Mythos52->value;
    public const MODEL_CLAUDE_MYTHOS_5_1 = ClaudeModel::Mythos51->value;
    public const MODEL_CLAUDE_MYTHOS_5 = ClaudeModel::Mythos5->value;
    public const MODEL_CLAUDE_OPUS_5_5 = ClaudeModel::Opus55->value;
    public const MODEL_CLAUDE_SONNET_5_5 = ClaudeModel::Sonnet55->value;
    public const MODEL_CLAUDE_HAIKU_5_5 = ClaudeModel::Haiku55->value;
    public const MODEL_CLAUDE_OPUS_5 = ClaudeModel::Opus5->value;
    public const MODEL_CLAUDE_SONNET_5 = ClaudeModel::Sonnet5->value;
    public const MODEL_CLAUDE_FABLE_5 = ClaudeModel::Fable5->value;
    public const MODEL_CLAUDE_HAIKU_4_5 = ClaudeModel::Haiku45->value;

    // Previous generations, still active.
    public const MODEL_CLAUDE_OPUS_4_8 = ClaudeModel::Opus48->value;
    public const MODEL_CLAUDE_OPUS_4_7 = ClaudeModel::Opus47->value;
    public const MODEL_CLAUDE_OPUS_4_6 = ClaudeModel::Opus46->value;
    public const MODEL_CLAUDE_SONNET_4_6 = ClaudeModel::Sonnet46->value;

    /**
     * The two levels Anthropic spells differently from the neutral scale.
     */
    private const NATIVE_EFFORT = [
        'extra-high' => 'xhigh',
        'maximum' => 'max',
    ];

    /**
     * Blocks the API wants back exactly as it sent them.
     */
    private const SENT_BACK_BLOCKS = ['thinking', 'redacted_thinking', 'compaction'];

    /**
     * Of those, the ones that carry the model's reasoning.
     */
    private const THINKING_BLOCKS = ['thinking', 'redacted_thinking'];

    private const THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01';

    protected ?int $lastRetryAfter = null;

    /**
     * @param string      $apiKey       Anthropic API key for authentication
     * @param string      $defaultModel Default model identifier
     * @param int         $defaultMaxTokens Default maximum tokens for responses
     * @param Effort|null $defaultEffort Extended-thinking effort when none is given per call
     * @param ThinkingReplay $thinkingReplay Whether thinking goes back to the API, for a caller whose history only grows
     * @param list<ContextEditInterface> $contextEdits Edits the API applies on its side, such as compaction
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_CLAUDE_SONNET_5_5,
        private readonly int $defaultMaxTokens = 4096,
        private readonly ?Effort $defaultEffort = null,
        private readonly ThinkingReplay $thinkingReplay = ThinkingReplay::Off,
        private readonly array $contextEdits = [],
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
        $content = $payload['content'] ?? [];

        foreach ($content as $block) {
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
            usage: $this->usageIncludingCompaction($payload['usage'] ?? []),
            stopReason: $payload['stop_reason'] ?? null,
            stoppedBecause: $this->stoppedBecauseOf($payload['stop_reason'] ?? null),
            replay: $this->hasBlockToSendBack($content) ? [$this->getName() => $content] : [],
        );
    }

    /**
     * Whether the content holds a block the API wants back verbatim: thinking, or a compaction summary.
     *
     * @param array<array<string, mixed>> $content
     */
    private function hasBlockToSendBack(array $content): bool
    {
        foreach ($content as $block) {
            if (in_array($block['type'] ?? null, self::SENT_BACK_BLOCKS, true)) {
                return true;
            }
        }

        return false;
    }

    /**
     * A compaction is billed as an iteration of its own, which the top-level counts leave out.
     *
     * @param array<string, mixed> $usage
     *
     * @return array<string, mixed>
     */
    private function usageIncludingCompaction(array $usage): array
    {
        $iterations = $usage['iterations'] ?? [];

        if (!is_array($iterations) || $iterations === []) {
            return $usage;
        }

        return array_replace($usage, [
            'input_tokens' => array_sum(array_column($iterations, 'input_tokens')),
            'output_tokens' => array_sum(array_column($iterations, 'output_tokens')),
        ]);
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
     * @return bool Always true; every Claude model takes a schema in `output_config.format`
     */
    public function supportsStructuredOutput(): bool
    {
        return true;
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
        $model = (string) ($options['model'] ?? $this->defaultModel);
        $payload = [
            'model' => $model,
            'max_tokens' => $options['maxTokens'] ?? $this->defaultMaxTokens,
        ];

        if (isset($options['temperature'])) {
            $this->assertAcceptsSampling($model);
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
                $this->assertCanForce($choice, $model);

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
            $payload = $this->withThinking($payload, $effort, $model);
        }

        if (isset($options['outputSchema'])) {
            $payload['output_config']['format'] = [
                'type' => 'json_schema',
                'schema' => (new OutputSchema())->forApi($options['outputSchema']),
            ];
        }

        $bindable = $this->thinksAdaptively($payload);
        $payload = $this->withConversation($payload, $messages, $options, $this->thinkingReplay->replaysWhen($bindable));

        if ($bindable) {
            $payload = $this->withThinkingBinding($payload);
        }

        return $this->withContextEdits($payload);
    }

    /**
     * The system prompt and messages, with each assistant turn as the API expects it back.
     *
     * Leading system messages make the system prompt. A later one stays where it is, on a model
     * that takes it there, so changing an instruction never rewrites the history already sent.
     *
     * @param array<string, mixed> $payload  The request so far
     * @param array<mixed>         $messages The caller's conversation
     * @param array<string, mixed> $options  The caller's request options
     * @param bool                 $replayThinking Whether thinking blocks go back
     *
     * @return array<string, mixed> The request with system and messages
     *
     * @throws ProviderException When a later system message sits where the API rejects one
     */
    private function withConversation(array $payload, array $messages, array $options, bool $replayThinking): array
    {
        $systemMessage = null;
        $apiMessages = [];
        $systemPositions = [];
        $inPlace = $this->acceptsMidConversationSystem((string) $payload['model']);

        foreach ($messages as $position => $message) {
            if (!$message instanceof Message) {
                continue;
            }

            if ($message->isSystem() && ($apiMessages === [] || !$inPlace)) {
                $systemMessage = $message->getText();

                continue;
            }

            if ($message->isSystem()) {
                $systemPositions[count($apiMessages)] = $position;
                $apiMessages[] = ['role' => 'system', 'content' => $message->getText()];

                continue;
            }

            $apiMessages[] = $this->convertMessage($message, $replayThinking);
        }

        $this->assertSystemMessagesPlaced($apiMessages, $systemPositions);
        $payload['messages'] = $apiMessages;

        if ($systemMessage === null) {
            return $payload;
        }

        $payload['system'] = ($options['cache'] ?? false) === true
            ? [['type' => 'text', 'text' => $systemMessage, 'cache_control' => ['type' => 'ephemeral']]]
            : $systemMessage;

        return $payload;
    }

    /**
     * Refuse a later system message the API would reject for where it sits.
     *
     * It must follow a user message, and be either the last message or come just before an assistant one.
     *
     * @param list<array<string, mixed>> $apiMessages     The messages as they will be sent
     * @param array<int, int|string>     $systemPositions Each later system message's index there, mapped to the caller's
     *
     * @throws ProviderException Naming the caller's position of the misplaced message
     */
    private function assertSystemMessagesPlaced(array $apiMessages, array $systemPositions): void
    {
        foreach ($systemPositions as $index => $position) {
            $previous = $apiMessages[$index - 1]['role'] ?? null;
            $next = $apiMessages[$index + 1]['role'] ?? 'assistant';

            if ($previous === 'user' && $next === 'assistant') {
                continue;
            }

            throw new ProviderException(
                sprintf(
                    'The system message at position %s must follow a user message and be the last message or come just before an assistant one. Append it after the next user message instead.',
                    $position,
                ),
                $this->getName(),
            );
        }
    }

    /**
     * Whether this request thinks adaptively, the one mode that accepts a binding.
     *
     * An omitted field means the model's own default, which is adaptive from Opus 5 on and for any
     * model we have not heard of.
     *
     * @param array<string, mixed> $payload The request so far
     */
    private function thinksAdaptively(array $payload): bool
    {
        if (isset($payload['thinking'])) {
            return ($payload['thinking']['type'] ?? null) === 'adaptive';
        }

        $model = (string) $payload['model'];

        if ($this->takesBudget($model)) {
            return false;
        }

        return $this->known($model)?->thinksByDefault() ?? true;
    }

    /**
     * Tell the API what to do with thinking a history edit invalidated, when thinking goes back.
     *
     * @param array<string, mixed> $payload A request that thinks adaptively
     *
     * @return array<string, mixed>
     */
    private function withThinkingBinding(array $payload): array
    {
        $behaviour = $this->thinkingReplay->prefixMismatchBehavior();

        if ($behaviour === null) {
            return $payload;
        }

        $payload['thinking'] = [
            'type' => 'adaptive',
            'block_binding' => ['prefix_mismatch_behavior' => $behaviour],
        ];

        return $payload;
    }

    /**
     * @param array<string, mixed> $payload The request so far
     *
     * @return array<string, mixed>
     */
    private function withContextEdits(array $payload): array
    {
        if ($this->contextEdits === []) {
            return $payload;
        }

        $payload['context_management'] = [
            'edits' => array_map(static fn (ContextEditInterface $edit): array => $edit->edit(), $this->contextEdits),
        ];

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
     * Haiku 4.5 and older want a token budget. Everything from 4.6 on wants adaptive thinking
     * plus an effort level. Models that cannot stop thinking never get a thinking block, and
     * "none" narrows to the shallowest level. Where thinking can be switched off, the block that
     * does it differs by model, so the model is asked rather than assumed.
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
            $payload['output_config'] = ['effort' => $this->nativeLevel($effort, $model)];

            return $payload;
        }

        if ($this->takesBudget($model)) {
            return $this->withBudget($payload, $effort);
        }

        if (!$effort->thinks()) {
            $switch = $this->offSwitchFor($model);

            if ($switch !== null) {
                $payload['thinking'] = ['type' => $switch];
            }

            return $payload;
        }

        $payload['thinking'] = ['type' => 'adaptive'];
        $payload['output_config'] = ['effort' => $this->nativeLevel($effort, $model)];

        return $payload;
    }

    /**
     * The block that turns thinking off for this model, or null to send nothing.
     *
     * Nothing is sent for a model that is off by default, and for one we have not heard of:
     * omitting the field is the one shape no model rejects.
     */
    private function offSwitchFor(string $model): ?string
    {
        $known = $this->known($model);

        if ($known === null || !$known->thinksByDefault()) {
            return null;
        }

        return $known->offSwitch();
    }

    /**
     * Refuse sampling parameters on the models that return 400 for them.
     *
     * Every model from 4.7 on rejects `temperature`; so does anything we have not heard of,
     * since the newest generations all do.
     *
     * @throws ProviderException When the model does not accept sampling parameters
     */
    private function assertAcceptsSampling(string $model): void
    {
        if ($this->known($model)?->acceptsSampling() ?? false) {
            return;
        }

        throw new ProviderException(
            sprintf('Model "%s" does not accept "temperature": the API returns 400 for sampling parameters. Remove it and let the model use its own default.', $model),
            $this->getName(),
        );
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
        $level = $effort->nearestOf($this->levelsFor($model))->value;

        return self::NATIVE_EFFORT[$level] ?? $level;
    }

    /**
     * What we know about this model, or null for an ID we have not shipped a case for yet.
     */
    private function known(string $model): ?ClaudeModel
    {
        return ClaudeModel::tryFrom($model);
    }

    /**
     * Fable and Mythos: thinking is always on and a "disabled" block is a 400.
     *
     * Unrecognised IDs in those families are matched by name, since a new Fable is far likelier
     * to keep the family's rules than to drop them.
     */
    private function thinksAlways(string $model): bool
    {
        $known = $this->known($model);

        if ($known !== null) {
            return $known->thinking() === ThinkingApi::AlwaysOn;
        }

        return preg_match('/claude-(fable|mythos)-/i', $model) === 1;
    }

    /**
     * Pre-4.6 models, which take a token budget and reject an effort level.
     *
     * Anything unrecognised is assumed to be newer, not older.
     */
    private function takesBudget(string $model): bool
    {
        $known = $this->known($model);

        if ($known !== null) {
            return $known->thinking() === ThinkingApi::Budget;
        }

        return preg_match('/claude-(haiku-4-5|3-|opus-4-[15]|sonnet-4-5|(opus|sonnet)-4-2025)/i', $model) === 1;
    }

    /**
     * The effort levels this model accepts. Unknown models are assumed to take the full scale.
     *
     * @return non-empty-list<Effort>
     */
    private function levelsFor(string $model): array
    {
        $levels = $this->known($model)?->effortLevels() ?? [];

        return $levels !== []
            ? $levels
            : [Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum];
    }

    /**
     * Whether a later system message can stay in place. Unknown IDs are assumed to take one, since
     * the newest generations all do.
     */
    private function acceptsMidConversationSystem(string $model): bool
    {
        return $this->known($model)?->acceptsMidConversationSystem() ?? true;
    }

    /**
     * Whether a forced tool choice is a 400 on this model. Unknown IDs are assumed to refuse,
     * since the newest generations all do.
     */
    private function refusesForcedTools(string $model): bool
    {
        return !($this->known($model)?->acceptsForcedTools() ?? false);
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
    private function convertMessage(Message $message, bool $replayThinking): array
    {
        $sentBack = $message->isAssistant() ? $this->contentToSendBack($message, $replayThinking) : [];

        if ($sentBack !== []) {
            return ['role' => 'assistant', 'content' => $sentBack];
        }

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
                    'input' => $toolCall->argumentsAsObject(),
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
     * The assistant turn exactly as the API returned it, less thinking when that is not going back.
     *
     * Decoding loses the difference between an empty object and an empty list, so tool inputs are
     * made objects again.
     *
     * @return list<array<string, mixed>> Empty when this provider left nothing on the message
     */
    private function contentToSendBack(Message $message, bool $replayThinking): array
    {
        $content = [];

        foreach ($message->replayFor($this->getName()) as $block) {
            if (!is_array($block) || (!$replayThinking && in_array($block['type'] ?? null, self::THINKING_BLOCKS, true))) {
                continue;
            }

            $content[] = ($block['type'] ?? null) === 'tool_use'
                ? array_replace($block, ['input' => (object) ($block['input'] ?? [])])
                : $block;
        }

        return $content;
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
            CURLOPT_HTTPHEADER => $this->headersFor($payload),
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
     * The HTTP headers for this request, with the betas its body relies on.
     *
     * @param array<string, mixed> $payload The request body
     *
     * @return list<string>
     */
    protected function headersFor(array $payload): array
    {
        $headers = [
            'Content-Type: application/json',
            'x-api-key: ' . $this->apiKey,
            'anthropic-version: ' . self::API_VERSION,
        ];
        $betas = $this->betasFor($payload);

        if ($betas !== []) {
            $headers[] = 'anthropic-beta: ' . implode(',', $betas);
        }

        return $headers;
    }

    /**
     * @param array<string, mixed> $payload The request body
     *
     * @return list<string>
     */
    private function betasFor(array $payload): array
    {
        $betas = [];

        if (isset($payload['thinking']['block_binding'])) {
            $betas[] = self::THINKING_BINDING_BETA;
        }

        foreach ($this->contextEdits as $edit) {
            $betas[] = $edit->beta();
        }

        if ($this->holdsCompaction($payload['messages'] ?? [])) {
            $betas[] = Compaction::BETA;
        }

        return array_values(array_unique($betas));
    }

    /**
     * Whether a compaction block is going back, so its beta goes with it even once this provider no longer compacts.
     *
     * @param array<array<string, mixed>> $messages
     */
    private function holdsCompaction(array $messages): bool
    {
        foreach ($messages as $message) {
            if (is_array($message['content'] ?? null) && in_array('compaction', array_column($message['content'], 'type'), true)) {
                return true;
            }
        }

        return false;
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
            CURLOPT_HTTPHEADER => [...$this->headersFor($payload), 'Accept: text/event-stream'],
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

    /**
     * Why the model stopped, in core's vocabulary.
     */
    private function stoppedBecauseOf(?string $stopReason): StopReason
    {
        return match ($stopReason) {
            'end_turn', 'stop_sequence' => StopReason::Stop,
            'max_tokens', 'model_context_window_exceeded' => StopReason::MaxTokens,
            'tool_use' => StopReason::ToolCalls,
            'refusal' => StopReason::Refused,
            default => StopReason::Other,
        };
    }
}
