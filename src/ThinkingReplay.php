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
 * Whether Claude's thinking goes back to the API on the next request, and what happens when it no longer can.
 *
 * Within a tool-use turn Anthropic wants the thinking blocks back. Each block is bound to the system
 * prompt, the tool set and every earlier message as they were when it was written, so a caller that
 * edits its history instead of appending to it invalidates the blocks that follow the edit.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/preserved-thinking
 */
enum ThinkingReplay: string
{
    /**
     * Thinking stays out of the history sent back. Safe for any caller.
     */
    case Off = 'off';

    /**
     * Thinking goes back, and a history edit that invalidated it fails the request with a 400.
     */
    case FailOnEdit = 'fail_on_edit';

    /**
     * Thinking goes back, and the API drops whatever a history edit invalidated instead of failing.
     */
    case DropOnEdit = 'drop_on_edit';

    /**
     * Whether thinking goes back on a request that can, or cannot, tell the API how to treat an edit.
     *
     * The binding needs adaptive thinking. Where it cannot be sent, dropping is only guaranteed by
     * leaving the thinking out.
     */
    public function replaysWhen(bool $bindable): bool
    {
        return match ($this) {
            self::Off => false,
            self::FailOnEdit => true,
            self::DropOnEdit => $bindable,
        };
    }

    /**
     * The `prefix_mismatch_behavior` this asks for, or null when nothing goes back.
     */
    public function prefixMismatchBehavior(): ?string
    {
        return match ($this) {
            self::Off => null,
            self::FailOnEdit => 'error',
            self::DropOnEdit => 'drop_block',
        };
    }
}
