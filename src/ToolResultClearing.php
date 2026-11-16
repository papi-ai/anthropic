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

/**
 * Clear old tool results on the server once the input passes a threshold, keeping the most recent ones.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/context-editing
 */
final class ToolResultClearing implements ContextEditInterface
{
    /**
     * @param int          $triggerTokens Input tokens at which clearing starts
     * @param int          $keep          How many of the latest tool uses keep their results
     * @param list<string> $excludeTools  Tools whose results are never cleared
     */
    public function __construct(
        public readonly int $triggerTokens = 100000,
        public readonly int $keep = 3,
        public readonly array $excludeTools = [],
    ) {
    }

    public function edit(): array
    {
        $edit = [
            'type' => 'clear_tool_uses_20250919',
            'trigger' => ['type' => 'input_tokens', 'value' => $this->triggerTokens],
            'keep' => ['type' => 'tool_uses', 'value' => $this->keep],
        ];

        if ($this->excludeTools !== []) {
            $edit['exclude_tools'] = $this->excludeTools;
        }

        return $edit;
    }

    public function beta(): string
    {
        return 'context-management-2025-06-27';
    }
}
