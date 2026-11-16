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

/**
 * Summarise the conversation on the server once its input passes a threshold.
 *
 * The response then opens with a compaction block, which must go back on every later request; the API
 * ignores whatever came before it.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/compaction-threshold
 */
final class Compaction implements ContextEditInterface
{
    public const BETA = 'compact-2026-01-12';
    public const MINIMUM_TRIGGER = 50000;

    /**
     * @param int         $triggerTokens Input tokens at which the conversation is summarised
     * @param string|null $instructions  A summarisation prompt replacing Anthropic's default one entirely
     *
     * @throws InvalidArgumentException When the trigger is below the API's minimum
     */
    public function __construct(
        public readonly int $triggerTokens = 150000,
        public readonly ?string $instructions = null,
    ) {
        if ($triggerTokens < self::MINIMUM_TRIGGER) {
            throw new InvalidArgumentException(sprintf('Compaction cannot trigger below %d input tokens; %d given.', self::MINIMUM_TRIGGER, $triggerTokens));
        }
    }

    public function edit(): array
    {
        $edit = [
            'type' => 'compact_20260112',
            'trigger' => ['type' => 'input_tokens', 'value' => $this->triggerTokens],
        ];

        if ($this->instructions !== null) {
            $edit['instructions'] = $this->instructions;
        }

        return $edit;
    }

    public function beta(): string
    {
        return self::BETA;
    }
}
