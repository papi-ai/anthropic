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
 * An edit the API applies to the conversation on its own side, which the history sent never shows.
 *
 * Server-side edits leave the caller's transcript untouched, so they never invalidate earlier thinking.
 */
interface ContextEditInterface
{
    /**
     * The entry for `context_management.edits`.
     *
     * @return array<string, mixed>
     */
    public function edit(): array;

    /**
     * The `anthropic-beta` value the edit needs.
     */
    public function beta(): string;
}
