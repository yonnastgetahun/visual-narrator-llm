# adult.vnpoverview.com Future Spec

## Purpose

This spec defines future work for `adult.vnpoverview.com` focused on reducing run cost without lowering output quality, reducing accepted vision coverage, or making descriptions less direct.

The adult demo should remain:

- explicit when the visuals justify it
- accessibility-focused rather than euphemistic
- neutral in tone rather than eroticized
- high quality on vision understanding

## Non-Negotiables

- Do not reduce the current vision quality bar as a cost shortcut.
- Do not reduce accepted vision coverage unless a separate spec explicitly changes that policy.
- Do not make descriptions more vague, more euphemistic, or less direct in order to save cost.
- Do not shift the narration style toward promotional or erotic copy.
- Keep the existing spend-protection and access-protection posture unless a later spec tightens it further.

## Cost Reduction Strategy

Cost reduction should come primarily from reuse, caching, and waste reduction.

### 1. Whole-Run Caching

If the same underlying source is submitted again with the same processing policy, the system should reuse existing outputs instead of recomputing the run.

Cached outputs should include:

- manifest
- per-gap narration text
- per-gap audio
- SRT
- mixed track

The run cache key should include:

- normalized source URL
- processing policy version
- model slug
- model version
- prompt version
- voice ID
- TTS model

### 2. Frame-Description Caching

Cache Replicate vision outputs at the frame level.

The cache should be keyed by:

- frame hash
- model slug
- model version
- prompt version

If a later run requests a frame that hashes identically to a previously described frame, skip the Replicate call and reuse the stored description.

This is a major cost lever because Replicate vision is the dominant cost driver.

### 3. TTS Audio Caching

Cache narration audio at the text-plus-voice level.

The cache key should include:

- final narration text
- voice ID
- TTS model
- voice settings version

If the same narration line is synthesized again with the same voice settings, reuse the prior audio payload instead of calling ElevenLabs again.

### 4. Failed-Run Waste Reduction

Reduce provider spend caused by bad sources and recoverable failures.

Future improvements should include:

- earlier rejection of unreadable or blocked sources
- stronger source validation before expensive downstream work
- narrow retries only for:
  - empty vision output
  - transient throttling
  - transient provider/server failures

Do not add broad, repeated retries for arbitrary failures.

### 5. Finished-Job Persistence and Retrieval

Completed runs should be recoverable without recomputation.

If a finished job is reopened, refreshed, or revisited, the system should serve stored outputs instead of regenerating them.

Persisted artifacts should include:

- manifest JSON
- SRT
- mixed narration track
- per-gap narration audio

### 6. Access Controls Instead of Quality Cuts

Control spend by limiting who can run the adult demo rather than weakening the run itself.

Future access-control options may include:

- password protection
- allowlisted users
- operator-issued access tokens
- restricted distribution

The age gate remains a UI gate, not a spend-control mechanism.

## Description Style Requirements

Adult descriptions should remain:

- direct
- explicit when visually justified
- accessibility-oriented
- neutral
- non-promotional
- non-sensational

Narration guidelines:

- name visible sexual actions directly when clearly visible
- name visible body positions directly when clearly visible
- mention visible clothing changes directly when relevant
- use anatomically specific terms when visually justified
- do not infer feelings, consent, or off-screen activity
- keep wording concise without becoming vague

The target style is explicit accessibility narration, not erotic narration.

## Implementation Priorities

Priority order:

1. whole-run cache
2. frame-description cache
3. TTS cache
4. finished-job persistence and retrieval
5. stronger access controls
6. additional early source-validation improvements

## Success Criteria

The future implementation should satisfy all of the following:

- repeated submissions of the same source reuse prior outputs whenever possible
- repeated frames across runs reuse prior vision descriptions
- repeated narration lines reuse prior TTS audio
- failed or blocked sources are rejected before expensive processing whenever possible
- adult descriptions remain explicit, neutral, and accessibility-focused
- per-run quality remains materially unchanged
- savings come from reuse and waste reduction rather than weaker output

## Current Direction

As of this spec:

- the adult demo already enforces spend guardrails
- the adult demo already uses direct and explicit accessibility-oriented prompting
- the next major efficiency milestone should be caching and output reuse
