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

use InvalidArgumentException;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Exception\ConfigurationException;
use PapiAI\Core\Factory\AbstractProviderFactory;
use PapiAI\Core\Factory\Setting;
use PapiAI\Core\Factory\SettingValues;

/**
 * Builds an AnthropicProvider from plain settings, so an application can select it by name.
 */
final class AnthropicProviderFactory extends AbstractProviderFactory
{
    public function name(): string
    {
        return 'anthropic';
    }

    public function package(): string
    {
        return 'papi-ai/anthropic';
    }

    public function settings(): array
    {
        return [
            new Setting('api_key', true, 'Anthropic API key'),
            new Setting('model', false, 'Default model, used when a request names none'),
            new Setting('max_tokens', false, 'Default output ceiling, used when a request sets none'),
            new Setting('effort', false, 'Default reasoning effort, used when a request sets none'),
            new Setting('replay_thinking', false, 'Send thinking back: "off", "fail_on_edit" or "drop_on_edit"'),
            new Setting('compact_at', false, 'Input tokens at which the server summarises the conversation'),
            new Setting('clear_tool_results_at', false, 'Input tokens at which the server clears old tool results'),
        ];
    }

    protected function build(SettingValues $values): ProviderInterface
    {
        return new AnthropicProvider(...$this->argumentsGiven([
            'apiKey' => $values->string('api_key'),
            'defaultModel' => $values->stringOrNull('model'),
            'defaultMaxTokens' => $values->intOrNull('max_tokens'),
            'defaultEffort' => $values->effort(),
            'thinkingReplay' => $this->thinkingReplay($values->stringOrNull('replay_thinking')),
            'contextEdits' => $this->contextEdits($values->intOrNull('clear_tool_results_at'), $values->intOrNull('compact_at')),
        ]));
    }

    /**
     * @throws ConfigurationException When the value is not one ThinkingReplay defines
     */
    private function thinkingReplay(?string $value): ?ThinkingReplay
    {
        if ($value === null) {
            return null;
        }

        return ThinkingReplay::tryFrom($value) ?? throw new ConfigurationException(sprintf(
            'Setting "replay_thinking" of provider "%s" must be one of %s; "%s" given.',
            $this->name(),
            implode(', ', array_map(static fn (ThinkingReplay $replay): string => '"' . $replay->value . '"', ThinkingReplay::cases())),
            $value,
        ));
    }

    /**
     * @return list<ContextEditInterface>
     */
    private function contextEdits(?int $clearToolResultsAt, ?int $compactAt): array
    {
        $edits = [];

        if ($clearToolResultsAt !== null) {
            $edits[] = new ToolResultClearing($clearToolResultsAt);
        }

        if ($compactAt !== null) {
            $edits[] = $this->compaction($compactAt);
        }

        return $edits;
    }

    /**
     * @throws ConfigurationException When the trigger is below the API's minimum
     */
    private function compaction(int $triggerTokens): Compaction
    {
        try {
            return new Compaction($triggerTokens);
        } catch (InvalidArgumentException $e) {
            throw new ConfigurationException(sprintf('Setting "compact_at" of provider "%s" is invalid: %s', $this->name(), $e->getMessage()), 0, $e);
        }
    }
}
