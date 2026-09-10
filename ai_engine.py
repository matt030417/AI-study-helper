from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "target_concepts": {"type": "array", "items": {"type": "string"}},
        "question_type": {"type": "string", "enum": ["concept", "calculation", "mixed"]},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "why_this_question": {"type": "string"},
    },
    "required": ["question", "target_concepts", "question_type", "difficulty", "why_this_question"],
    "additionalProperties": False,
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "weak_concepts": {"type": "array", "items": {"type": "string"}},
        "error_type": {
            "type": "string",
            "enum": [
                "none",
                "concept_gap",
                "formula_choice",
                "formula_application",
                "calculation",
                "unit",
                "condition_interpretation",
                "terminology",
                "insufficient_answer",
            ],
        },
        "feedback": {"type": "string"},
        "ideal_answer": {"type": "string"},
    },
    "required": [
        "score",
        "verdict",
        "strengths",
        "missing_points",
        "weak_concepts",
        "error_type",
        "feedback",
        "ideal_answer",
    ],
    "additionalProperties": False,
}

PROBLEM_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "problem_type": {"type": "string", "enum": ["concept", "calculation"]},
        "concept": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "answer_format": {"type": "string"},
        "grading_mode": {"type": "string", "enum": ["llm", "numeric"]},
        "numeric_answer": {"type": ["number", "null"]},
        "numeric_tolerance": {"type": ["number", "null"]},
        "expected_unit": {"type": ["string", "null"]},
        "reference_answer": {"type": "string"},
        "solution": {"type": "string"},
    },
    "required": [
        "problem",
        "problem_type",
        "concept",
        "difficulty",
        "answer_format",
        "grading_mode",
        "numeric_answer",
        "numeric_tolerance",
        "expected_unit",
        "reference_answer",
        "solution",
    ],
    "additionalProperties": False,
}

PRACTICE_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "error_type": {
            "type": "string",
            "enum": [
                "none",
                "concept_gap",
                "formula_choice",
                "formula_application",
                "calculation",
                "unit",
                "condition_interpretation",
                "terminology",
                "insufficient_answer",
            ],
        },
        "reason": {"type": "string"},
        "feedback": {"type": "string"},
        "weak_concepts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_correct", "score", "error_type", "reason", "feedback", "weak_concepts"],
    "additionalProperties": False,
}

NUMERIC_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "error_type": {
            "type": "string",
            "enum": [
                "formula_choice",
                "formula_application",
                "calculation",
                "unit",
                "condition_interpretation",
                "concept_gap",
                "unknown",
            ],
        },
        "reason": {"type": "string"},
        "feedback": {"type": "string"},
        "weak_concepts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["error_type", "reason", "feedback", "weak_concepts"],
    "additionalProperties": False,
}


class AIEngine:
    def __init__(self, api_key: str, model: str = "gpt-5.6-luna"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _structured(self, *, name: str, schema: dict, instructions: str, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
            reasoning={"effort": "low"},
            store=False,
        )
        return json.loads(response.output_text)

    def generate_oral_question(
        self,
        context: str,
        topic: str,
        mode: str,
        difficulty: str,
        weak_hint: str = "",
    ) -> dict:
        instructions = """당신은 대학 수업의 구술시험 출제자다.
제공된 강의자료와 기출문제 맥락 안에서만 질문을 만든다.
기출문제는 그대로 복사하기보다 같은 핵심 개념/출제 방식의 새로운 질문을 만든다.
학생이 단순 암기보다 이해와 적용을 보여줄 수 있는 질문을 선호한다.
맥락에 없는 세부 사실을 임의로 추가하지 않는다."""

        prompt = f"""[학습 범위]
{topic or "자료 전체"}

[시험 모드]
{mode}

[난이도]
{difficulty}

[특히 확인할 취약 개념]
{weak_hint or "없음"}

[강의자료/기출 맥락]
{context}

위 정보를 이용해 구술시험 질문 1개를 생성하라."""
        return self._structured(
            name="oral_question",
            schema=QUESTION_SCHEMA,
            instructions=instructions,
            prompt=prompt,
        )

    def evaluate_oral_answer(self, question: str, answer: str, context: str) -> dict:
        instructions = """당신은 엄격하지만 교육적인 대학 구술시험 채점자다.
학생 답변을 제공된 자료와 질문 의도에 근거해 평가한다.
단순 표현 차이는 오답 처리하지 않는다.
부분적으로 맞으면 partial로 평가하고 빠진 핵심을 명확히 짚는다.
학생이 틀린 이유를 가능한 한 구체적인 오류 유형으로 분류한다.
이상적인 답안은 학습용으로 간결하고 정확하게 작성한다."""

        prompt = f"""[질문]
{question}

[학생 답변]
{answer}

[관련 강의자료]
{context}

학생의 이해도를 채점하고 취약 개념을 찾아라."""
        return self._structured(
            name="oral_evaluation",
            schema=EVALUATION_SCHEMA,
            instructions=instructions,
            prompt=prompt,
        )

    def generate_practice_problem(
        self,
        context: str,
        weak_concept: str,
        desired_type: str,
        difficulty: str,
    ) -> dict:
        instructions = """당신은 대학 시험 대비 맞춤 문제 출제자다.
제공된 강의자료와 기출 스타일을 참고하여 취약 개념을 보완하는 새로운 문제 1개를 만든다.
개념 문제와 계산 문제 모두 가능하다.
계산 문제에서 최종 답이 하나의 수치로 안정적으로 채점 가능하면 grading_mode=numeric으로 하고,
numeric_answer와 합리적인 절대오차 tolerance, expected_unit을 반드시 제공한다.
그렇지 않으면 grading_mode=llm으로 한다.
문제 안에 필요한 상수/수치가 자료에 없으면 문제 자체에서 제공한다.
기출문제 문장을 그대로 복제하지 않는다."""

        prompt = f"""[취약 개념]
{weak_concept}

[원하는 문제 유형]
{desired_type}

[난이도]
{difficulty}

[관련 강의자료/기출]
{context}

취약 개념을 실제로 확인할 수 있는 문제를 생성하라."""
        return self._structured(
            name="practice_problem",
            schema=PROBLEM_SCHEMA,
            instructions=instructions,
            prompt=prompt,
        )

    def evaluate_practice_answer(
        self,
        problem: dict,
        student_answer: str,
        student_work: str,
        context: str,
    ) -> dict:
        instructions = """당신은 학습용 문제 채점자다.
정답 여부뿐 아니라 왜 맞거나 틀렸는지 설명한다.
표현만 다른 동치 답은 정답으로 인정한다.
오답이면 오류 원인을 분류하고 다음에 무엇을 확인해야 하는지 구체적으로 피드백한다."""

        prompt = f"""[문제]
{problem['problem']}

[기준 답안]
{problem['reference_answer']}

[기준 풀이]
{problem['solution']}

[학생 답]
{student_answer}

[학생 풀이]
{student_work or "(풀이 미입력)"}

[관련 자료]
{context}

학생 답을 평가하라."""
        return self._structured(
            name="practice_evaluation",
            schema=PRACTICE_EVAL_SCHEMA,
            instructions=instructions,
            prompt=prompt,
        )

    def explain_numeric_error(
        self,
        problem: dict,
        student_answer: str,
        student_work: str,
        context: str,
    ) -> dict:
        instructions = """당신은 계산 문제 오답 코치다.
정답 판정은 이미 외부 계산 로직에서 완료되었다.
당신의 역할은 학생의 풀이와 기준 풀이를 비교해 오답의 가장 가능성 높은 원인을 설명하는 것이다.
풀이가 없으면 원인을 단정하지 말고 확인해야 할 지점을 제시한다."""

        prompt = f"""[문제]
{problem['problem']}

[정답]
{problem['numeric_answer']} {problem['expected_unit'] or ''}

[기준 풀이]
{problem['solution']}

[학생 최종 답]
{student_answer}

[학생 풀이]
{student_work or "(풀이 미입력)"}

[관련 자료]
{context}

오답 원인과 피드백을 작성하라."""
        return self._structured(
            name="numeric_feedback",
            schema=NUMERIC_FEEDBACK_SCHEMA,
            instructions=instructions,
            prompt=prompt,
        )
