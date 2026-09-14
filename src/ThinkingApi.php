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
 * The shape a Claude generation accepts for thinking, which decides how effort is sent.
 *
 * Three generations, three shapes. Sending the wrong one is a 400, not a downgrade.
 */
enum ThinkingApi
{
    /**
     * A token budget carved out of max_tokens: `thinking: {type: enabled, budget_tokens}`.
     * Haiku 4.5 and everything before 4.6. Rejects an effort level.
     */
    case Budget;

    /**
     * `thinking: {type: adaptive}` plus `output_config.effort`. 4.6 onwards. Rejects a budget.
     */
    case Adaptive;

    /**
     * Thinking cannot be switched off and a `disabled` block is a 400, so no thinking block is
     * ever sent; only `output_config.effort`. Fable and Mythos.
     */
    case AlwaysOn;
}
