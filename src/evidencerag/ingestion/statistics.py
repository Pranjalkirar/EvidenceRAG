"""Basic dataset statistics computed from the normalized internal
representation (never fabricated — always derived from actual `Paper`
objects passed in).

Usage:
    stats = compute_statistics(papers)
    print(stats.format_report())
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evidencerag.ingestion.schema import Paper


@dataclass
class SplitStatistics:
    split: str
    num_papers: int = 0
    num_questions: int = 0
    num_answers: int = 0
    num_evidence_annotations: int = 0
    num_unanswerable_answers: int = 0
    num_sections: int = 0
    num_paragraphs: int = 0
    num_figures_and_tables: int = 0
    num_evidence_resolved: int = 0
    num_evidence_unresolved: int = 0


@dataclass
class DatasetStatistics:
    by_split: dict[str, SplitStatistics] = field(default_factory=dict)

    @property
    def total_papers(self) -> int:
        return sum(s.num_papers for s in self.by_split.values())

    @property
    def total_questions(self) -> int:
        return sum(s.num_questions for s in self.by_split.values())

    @property
    def total_answers(self) -> int:
        return sum(s.num_answers for s in self.by_split.values())

    @property
    def total_evidence_annotations(self) -> int:
        return sum(s.num_evidence_annotations for s in self.by_split.values())

    def format_report(self) -> str:
        lines = ["QASPER dataset statistics", "=" * 40]
        for split_name, s in sorted(self.by_split.items()):
            avg_sections = s.num_sections / s.num_papers if s.num_papers else 0.0
            avg_paragraphs = s.num_paragraphs / s.num_papers if s.num_papers else 0.0
            resolution_rate = (
                s.num_evidence_resolved / s.num_evidence_annotations
                if s.num_evidence_annotations
                else 0.0
            )
            lines.append(f"\n[{split_name}]")
            lines.append(f"  papers:                 {s.num_papers}")
            lines.append(f"  questions:               {s.num_questions}")
            lines.append(f"  answers:                 {s.num_answers}")
            lines.append(f"  unanswerable answers:    {s.num_unanswerable_answers}")
            lines.append(f"  evidence annotations:    {s.num_evidence_annotations}")
            lines.append(
                f"  evidence resolved/unresolved: {s.num_evidence_resolved}/{s.num_evidence_unresolved} "
                f"({resolution_rate:.1%} resolved)"
            )
            lines.append(f"  figures/tables:          {s.num_figures_and_tables}")
            lines.append(f"  avg sections/paper:      {avg_sections:.1f}")
            lines.append(f"  avg paragraphs/paper:    {avg_paragraphs:.1f}")

        lines.append("\n[total]")
        lines.append(f"  papers:    {self.total_papers}")
        lines.append(f"  questions: {self.total_questions}")
        lines.append(f"  answers:   {self.total_answers}")
        lines.append(f"  evidence:  {self.total_evidence_annotations}")
        return "\n".join(lines)


def compute_statistics(papers: list[Paper]) -> DatasetStatistics:
    """Compute statistics purely from the given `Paper` objects.

    Nothing here is estimated or hardcoded — every number comes from
    counting fields on the objects actually passed in.
    """
    by_split: dict[str, SplitStatistics] = {}

    for paper in papers:
        s = by_split.setdefault(paper.split, SplitStatistics(split=paper.split))
        s.num_papers += 1
        s.num_sections += len(paper.sections)
        s.num_paragraphs += sum(len(sec.paragraphs) for sec in paper.sections)
        s.num_figures_and_tables += len(paper.figures_and_tables)
        s.num_questions += len(paper.questions)

        for question in paper.questions:
            for answer in question.answers:
                s.num_answers += 1
                if answer.unanswerable:
                    s.num_unanswerable_answers += 1
                s.num_evidence_annotations += len(answer.evidence)
                for evidence in answer.evidence:
                    if evidence.resolved:
                        s.num_evidence_resolved += 1
                    else:
                        s.num_evidence_unresolved += 1

    return DatasetStatistics(by_split=by_split)
