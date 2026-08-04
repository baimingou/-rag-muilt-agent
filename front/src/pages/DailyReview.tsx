import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { CheckCircle2, ChevronDown, ChevronRight, GraduationCap, RefreshCw, Sparkles, XCircle } from 'lucide-react'
import { reviewApi } from '../api/review'
import type { ReviewItem, ReviewQuestion } from '../types/api'
import EmptyState from '../components/common/EmptyState'

const FALLBACK_CHOICES = ['不太确定', '需要复习', '基本掌握', '完全理解']

function isFallbackQuestion(question: ReviewQuestion | undefined) {
  if (!question) return false
  return question.question === '请回顾这篇笔记的主要内容' &&
    question.choices.length === FALLBACK_CHOICES.length &&
    question.choices.every((choice, index) => choice === FALLBACK_CHOICES[index])
}

export default function DailyReview() {
  const { t } = useTranslation()
  const [items, setItems] = useState<ReviewItem[]>([])
  const [loading, setLoading] = useState(true)
  const [completedIds, setCompletedIds] = useState<string[]>([])
  const [quizNotes, setQuizNotes] = useState<Record<string, ReviewQuestion>>({})
  const [expandedNotes, setExpandedNotes] = useState<string[]>([])
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, string | null>>({})
  const [showResults, setShowResults] = useState<Record<string, boolean>>({})
  const [questionLoadingNoteId, setQuestionLoadingNoteId] = useState<string | null>(null)
  const [backfilling, setBackfilling] = useState(false)

  const loadReviews = async () => {
    setLoading(true)
    try {
      const data = await reviewApi.today()
      setItems(data.reviews || [])
      setCompletedIds([])
    } catch {
      toast.error('加载复习内容失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadReviews()
  }, [])

  const pendingItems = items.filter((item) => !completedIds.includes(item.note_id))
  const doneCount = completedIds.length
  const totalCount = items.length

  const toggleExpanded = (noteId: string) => {
    setExpandedNotes((prev) =>
      prev.includes(noteId) ? prev.filter((id) => id !== noteId) : [...prev, noteId]
    )
  }

  const handleBackfill = async () => {
    setBackfilling(true)
    try {
      const result = await reviewApi.backfill()
      toast.success(result.message || '历史复习记录已补建')
      await loadReviews()
    } catch {
      toast.error('补建历史复习记录失败')
    } finally {
      setBackfilling(false)
    }
  }

  const handleStartQuiz = async (noteId: string) => {
    toggleExpanded(noteId)
    if (quizNotes[noteId]) {
      return
    }

    setQuestionLoadingNoteId(noteId)
    try {
      const q = await reviewApi.getQuestion(noteId)
      if (q) {
        setQuizNotes((prev) => ({ ...prev, [noteId]: q }))
      }
    } catch {
      toast.error('获取题目失败')
    } finally {
      setQuestionLoadingNoteId(null)
    }
  }

  const handleRegenerate = async (noteId: string) => {
    setSelectedAnswers((prev) => ({ ...prev, [noteId]: null }))
    setShowResults((prev) => ({ ...prev, [noteId]: false }))
    setQuestionLoadingNoteId(noteId)
    try {
      const q = await reviewApi.getQuestion(noteId)
      if (q) {
        setQuizNotes((prev) => ({ ...prev, [noteId]: q }))
      }
    } catch {
      toast.error('重新生成题目失败')
    } finally {
      setQuestionLoadingNoteId(null)
    }
  }

  const handleAnswer = (noteId: string, answer: string) => {
    setSelectedAnswers((prev) => ({ ...prev, [noteId]: answer }))
    setShowResults((prev) => ({ ...prev, [noteId]: true }))
  }

  const handleMarkDone = async (noteId: string) => {
    try {
      await reviewApi.markDone(noteId)
      setCompletedIds((prev) => [...prev, noteId])
      toast.success('已标记为完成回顾')
    } catch {
      toast.error('标记回顾失败')
    }
  }

  return (
    <div className="max-w-4xl mx-auto py-8 px-6">
      <div className="flex items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="font-heading text-xl font-semibold text-[var(--color-text)]">{t('review.title')}</h1>
          {!loading && totalCount > 0 && (
            <p className="text-sm text-[var(--color-text-tertiary)] mt-1">
              今天共 {totalCount} 条待回顾内容，建议先快速浏览摘要，再按需做 AI 抽查。
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void handleBackfill()}
            disabled={backfilling}
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-md border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-secondary)] disabled:opacity-50 transition-colors"
          >
            <RefreshCw size={14} className={backfilling ? 'animate-spin' : ''} />
            补建历史回顾
          </button>
          {!loading && totalCount > 0 && (
            <span className="text-xs text-[var(--color-text-tertiary)]">{t('review.progress')}: {doneCount}/{totalCount}</span>
          )}
        </div>
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-40 bg-[var(--color-bg-tertiary)] rounded-lg animate-pulse" />
          ))}
        </div>
      ) : totalCount === 0 ? (
        <EmptyState
          icon={<GraduationCap size={48} />}
          message={t('review.empty')}
          action={
            <button
              onClick={() => void handleBackfill()}
              disabled={backfilling}
              className="mt-2 flex items-center gap-2 px-4 py-2 text-sm rounded-md bg-[var(--color-accent)] text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <RefreshCw size={14} className={backfilling ? 'animate-spin' : ''} />
              补建历史回顾
            </button>
          }
        />
      ) : pendingItems.length === 0 ? (
        <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-8 text-center">
          <GraduationCap size={48} className="mx-auto mb-4 text-[var(--color-success)]" />
          <p className="text-base font-medium text-[var(--color-text)] mb-2">{t('review.allDone')}</p>
          <p className="text-sm text-[var(--color-text-tertiary)]">{t('review.progress')}: {doneCount}/{totalCount}</p>
        </div>
      ) : (
        <div className="space-y-4">
          {pendingItems.map((item, index) => {
            const isExpanded = expandedNotes.includes(item.note_id)
            const question = quizNotes[item.note_id]
            const fallbackQuestion = isFallbackQuestion(question)
            const selectedAnswer = selectedAnswers[item.note_id]
            const showResult = showResults[item.note_id]
            const isQuestionLoading = questionLoadingNoteId === item.note_id
            const isCorrect = selectedAnswer != null && selectedAnswer === question?.answer

            return (
              <div key={item.review_id} className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--color-bg-secondary)] text-[var(--color-text-tertiary)]">
                        第 {index + 1} 条
                      </span>
                      <span className="text-xs text-[var(--color-text-tertiary)]">
                        今日待复习 | {item.review_count || 0} 次回顾
                      </span>
                    </div>
                    <h3 className="text-base font-medium text-[var(--color-text)] mb-2">{item.title}</h3>
                    <p className="text-sm text-[var(--color-text-secondary)] leading-6 whitespace-pre-wrap">
                      {item.content_preview || '这篇笔记暂无摘要内容，可直接标记完成或使用 AI 抽查。'}
                    </p>
                  </div>
                  <button
                    onClick={() => void handleMarkDone(item.note_id)}
                    className="shrink-0 flex items-center gap-2 px-4 py-2 text-sm rounded-md bg-[var(--color-success)] text-white hover:bg-green-700 transition-colors"
                  >
                    <CheckCircle2 size={16} />
                    标记已回顾
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-2 mt-4">
                  <button
                    onClick={() => void handleStartQuiz(item.note_id)}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-md border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-secondary)] transition-colors"
                  >
                    <Sparkles size={14} />
                    AI 抽查
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                  {item.category && (
                    <span className="text-xs px-2 py-1 rounded-full bg-[var(--color-accent-bg)] text-[var(--color-accent)]">
                      {item.category}
                    </span>
                  )}
                  {(item.tags || []).slice(0, 3).map((tag) => (
                    <span key={tag} className="text-xs px-2 py-1 rounded-full bg-[var(--color-bg-secondary)] text-[var(--color-text-tertiary)]">
                      #{tag}
                    </span>
                  ))}
                </div>

                {isExpanded && (
                  <div className="mt-5 pt-5 border-t border-[var(--color-border)]">
                    {isQuestionLoading ? (
                      <div className="space-y-3">
                        <div className="h-4 w-1/2 bg-[var(--color-bg-tertiary)] rounded animate-pulse" />
                        {[1, 2, 3, 4].map((i) => (
                          <div key={i} className="h-10 bg-[var(--color-bg-tertiary)] rounded animate-pulse" />
                        ))}
                      </div>
                    ) : fallbackQuestion ? (
                      <div className="space-y-3">
                        <div className="px-4 py-3 rounded-md bg-[var(--color-bg-secondary)] text-sm text-[var(--color-text-secondary)] leading-6">
                          这次 AI 没有生成有效的内容题，建议先根据上面的摘要回忆要点，再直接标记完成。需要的话可以重试一次抽查。
                        </div>
                        <button
                          onClick={() => void handleRegenerate(item.note_id)}
                          className="flex items-center gap-2 px-3 py-1.5 text-xs rounded-md border border-[var(--color-border)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-secondary)] transition-colors"
                        >
                          <RefreshCw size={12} />
                          重新生成题目
                        </button>
                      </div>
                    ) : question ? (
                      <div className="space-y-4">
                        <h4 className="text-sm font-medium text-[var(--color-text)]">{question.question}</h4>
                        <div className="space-y-2">
                          {question.choices.map((opt, choiceIndex) => {
                            const isSelected = selectedAnswer === opt
                            const isCorrectAnswer = opt === question.answer
                            let className = 'w-full text-left px-4 py-3 rounded-md border text-sm transition-colors '

                            if (!showResult) {
                              className += 'border-[var(--color-border)] hover:border-[var(--color-accent)] cursor-pointer'
                            } else if (isCorrectAnswer) {
                              className += 'border-[var(--color-success)] bg-[var(--color-success-bg)] text-[var(--color-success)]'
                            } else if (isSelected) {
                              className += 'border-[var(--color-danger)] bg-[var(--color-danger-bg)] text-[var(--color-danger)]'
                            } else {
                              className += 'border-[var(--color-border)] opacity-60'
                            }

                            return (
                              <button
                                key={opt}
                                className={className}
                                onClick={() => !showResult && handleAnswer(item.note_id, opt)}
                                disabled={showResult}
                              >
                                <span className="text-xs text-[var(--color-text-tertiary)] mr-2">
                                  {String.fromCharCode(65 + choiceIndex)}.
                                </span>
                                {opt}
                                {showResult && isCorrectAnswer && <CheckCircle2 size={14} className="inline ml-2" />}
                                {showResult && isSelected && !isCorrectAnswer && <XCircle size={14} className="inline ml-2" />}
                              </button>
                            )
                          })}
                        </div>

                        {showResult && (
                          <div className={`px-4 py-3 rounded-md text-sm ${isCorrect ? 'bg-[var(--color-success-bg)] text-[var(--color-success)]' : 'bg-[var(--color-danger-bg)] text-[var(--color-danger)]'}`}>
                            {isCorrect ? t('review.correct') : t('review.wrong')}
                          </div>
                        )}

                        <button
                          onClick={() => void handleRegenerate(item.note_id)}
                          className="flex items-center gap-2 px-3 py-1.5 text-xs rounded-md border border-[var(--color-border)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-secondary)] transition-colors"
                        >
                          <RefreshCw size={12} />
                          {t('review.regenerate')}
                        </button>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
