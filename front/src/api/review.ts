import client from './client'
import { endpoints } from './endpoints'
import type { ApiResponse, ReviewListData, ReviewQuestion } from '../types/api'

export const reviewApi = {
  today: async () => {
    const res = await client.get<ApiResponse<ReviewListData>>(endpoints.reviewToday)
    return res.data.data
  },

  backfill: async () => {
    const res = await client.post<ApiResponse<{ success: boolean; message: string; created_count: number; total_notes: number }>>(endpoints.reviewBackfill)
    return res.data.data
  },

  markDone: async (noteId: string) => {
    const res = await client.post<ApiResponse<null>>(endpoints.reviewDone(noteId))
    return res.data
  },

  getQuestion: async (noteId: string) => {
    const res = await client.get<ApiResponse<ReviewQuestion>>(endpoints.reviewQuestion(noteId))
    return res.data.data
  },
}
