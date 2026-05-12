import unittest
from pathlib import Path
from unittest.mock import patch

from vn.ad import (
    AudioGapFitMetrics,
    build_audio_gap_fit_summary,
    generate_gap_aware_ad_narration,
    target_words,
)
from vn.compliance import ComplianceCriterion, ComplianceReport
from vn.output import render_compliance_text


class AudioGapFitTests(unittest.TestCase):
    def test_target_words_respects_floor_and_cap(self) -> None:
        self.assertEqual(target_words(0.5), 5)
        self.assertEqual(target_words(4.0), 10)
        self.assertEqual(target_words(20.0), 30)

    def test_audio_gap_fit_summary_zero_safe(self) -> None:
        summary = build_audio_gap_fit_summary([])
        self.assertEqual(summary["narrations_evaluated"], 0)
        self.assertEqual(summary["overrun_rate"], 0.0)
        self.assertEqual(summary["truncation_count"], 0)

    @patch("vn.ad._log_audio_gap_fit")
    @patch("vn.ad.audio_duration_sec", side_effect=[4.20, 3.70])
    @patch("vn.ad.synthesize_speech_bytes", side_effect=[(b"first", 42), (b"second", 30)])
    @patch(
        "vn.ad._describe_frame_for_ad",
        side_effect=[("A longer first pass.", "gpt-4o"), ("Short retry.", "gpt-4o")],
    )
    def test_retries_before_fit(
        self,
        describe_mock,
        synth_mock,
        duration_mock,
        log_mock,
    ) -> None:
        result = generate_gap_aware_ad_narration(Path("/tmp/frame.jpg"), 4.0, "voice-1")

        self.assertEqual(result.description, "Short retry.")
        self.assertEqual(result.audio_fit.retries, 1)
        self.assertFalse(result.audio_fit.truncated)
        self.assertEqual(result.audio_fit.attempt_word_limits, (10, 7))
        self.assertEqual(result.audio_fit.overrun_attempts, 1)
        self.assertEqual(describe_mock.call_count, 2)
        self.assertEqual(synth_mock.call_count, 2)
        self.assertEqual(duration_mock.call_count, 2)
        log_mock.assert_called_once()

    @patch("vn.ad._log_audio_gap_fit")
    @patch("vn.ad._truncate_to_fit", return_value=(b"trimmed", 3.60))
    @patch("vn.ad.audio_duration_sec", side_effect=[4.30, 4.10, 3.95])
    @patch(
        "vn.ad.synthesize_speech_bytes",
        side_effect=[(b"one", 40), (b"two", 31), (b"three", 24)],
    )
    @patch(
        "vn.ad._describe_frame_for_ad",
        side_effect=[
            ("Attempt one.", "gpt-4o"),
            ("Attempt two.", "gpt-4o"),
            ("Attempt three.", "gpt-4o"),
        ],
    )
    def test_truncates_only_after_two_retries(
        self,
        describe_mock,
        synth_mock,
        duration_mock,
        truncate_mock,
        log_mock,
    ) -> None:
        result = generate_gap_aware_ad_narration(Path("/tmp/frame.jpg"), 4.0, "voice-1")

        self.assertEqual(result.description, "Attempt three.")
        self.assertEqual(result.audio_fit.retries, 2)
        self.assertTrue(result.audio_fit.truncated)
        self.assertEqual(result.audio_fit.overrun_attempts, 3)
        self.assertEqual(result.audio_fit.attempt_word_limits, (10, 7, 4))
        truncate_mock.assert_called_once_with(b"three", 4.0)
        self.assertEqual(describe_mock.call_count, 3)
        self.assertEqual(synth_mock.call_count, 3)
        self.assertEqual(duration_mock.call_count, 3)
        log_mock.assert_called_once()

    def test_summary_counts_overruns_and_truncations(self) -> None:
        metrics = [
            AudioGapFitMetrics(
                gap_sec=4.0,
                audio_sec=3.2,
                fit=True,
                retries=0,
                truncated=False,
                fit_ratio=0.8,
                overrun_attempts=0,
                word_limit=10,
                max_allowed_sec=3.85,
            ),
            AudioGapFitMetrics(
                gap_sec=4.0,
                audio_sec=3.6,
                fit=True,
                retries=2,
                truncated=True,
                fit_ratio=0.9,
                overrun_attempts=3,
                word_limit=4,
                max_allowed_sec=3.85,
            ),
        ]
        summary = build_audio_gap_fit_summary(metrics)

        self.assertEqual(summary["narrations_evaluated"], 2)
        self.assertEqual(summary["overrun_count"], 1)
        self.assertEqual(summary["overrun_rate"], 0.5)
        self.assertEqual(summary["retry_count"], 2)
        self.assertEqual(summary["truncation_count"], 1)
        self.assertEqual(summary["average_fit_ratio"], 0.85)

    def test_compliance_text_includes_audio_gap_fit_summary(self) -> None:
        report = ComplianceReport(
            score=100,
            wcag_level="AA",
            criteria={
                "wcag_1_2_3": ComplianceCriterion(
                    passed=True,
                    level="A",
                    description="criterion",
                    metric={"narration_gaps": 1},
                )
            },
            gaps=[],
            recommendations=[],
            total_duration_sec=12.0,
            coverage_percent=25.0,
            max_unbroken_speech_sec=8.0,
            audio_gap_fit={
                "narrations_evaluated": 2,
                "overrun_count": 1,
                "overrun_rate": 0.5,
                "retry_count": 2,
                "truncation_count": 1,
                "average_fit_ratio": 0.85,
                "buffer_sec": 0.15,
            },
        )

        rendered = render_compliance_text(report)
        self.assertIn("Audio gap fit:", rendered)
        self.assertIn("Overrun rate: 50.0% (1/2)", rendered)
        self.assertIn("Retries: 2 | Truncations: 1", rendered)


if __name__ == "__main__":
    unittest.main()
