import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import * as api from '../../api/companyExposures';
import { renderWithProviders } from '../../test/renderWithProviders';
import ExposureResearchWorkspace, { PREVIEW_REFRESH_MS } from './ExposureResearchWorkspace';
import { shadowCompletedJob, shadowPreview } from './fixtures';

vi.mock('../../api/companyExposures', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    requestExposureResearch: vi.fn(),
    getResearchJob: vi.fn(),
    getResearchJobPreview: vi.fn(),
  };
});

describe('ExposureResearchWorkspace', () => {
  beforeEach(() => vi.clearAllMocks());

  it('requests research and shows the job-scoped shadow preview', async () => {
    api.requestExposureResearch.mockResolvedValue({
      job_id: shadowCompletedJob.job_id, created: true, state: 'queued', dispatch: 'queued',
    });
    api.getResearchJob.mockResolvedValue(shadowCompletedJob);
    api.getResearchJobPreview.mockResolvedValue(shadowPreview);
    renderWithProviders(<ExposureResearchWorkspace />);

    expect(screen.queryByRole('button', { name: /request research/i })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/admin key/i), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));
    fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'exmp' } });
    fireEvent.change(screen.getByLabelText(/economic theme id/i), {
      target: { value: shadowCompletedJob.economic_theme_id },
    });
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));

    await waitFor(() => expect(api.requestExposureResearch).toHaveBeenCalled());
    const [key, body] = api.requestExposureResearch.mock.calls[0];
    expect(key).toBe('secret');
    expect(body).toMatchObject({
      kind: 'verify', symbol: 'EXMP', economicThemeId: shadowCompletedJob.economic_theme_id,
    });
    expect(body.idempotencyKey).toMatch(/^ui-/);
    expect(await screen.findByText(/shadow preview/i)).toBeVisible();
    expect(api.getResearchJobPreview).toHaveBeenCalledWith('secret', shadowCompletedJob.job_id);
    expect(screen.getByText('The ET-9000 supports HBM testing.')).toBeVisible();
  });

  it('refreshes the preview so live freshness and holds stay current', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      api.requestExposureResearch.mockResolvedValue({
        job_id: shadowCompletedJob.job_id, created: true, state: 'queued', dispatch: 'queued',
      });
      api.getResearchJob.mockResolvedValue(shadowCompletedJob);
      api.getResearchJobPreview.mockResolvedValue(shadowPreview);
      renderWithProviders(<ExposureResearchWorkspace />);
      fireEvent.change(screen.getByLabelText(/admin key/i), { target: { value: 'secret' } });
      fireEvent.click(screen.getByRole('button', { name: /unlock/i }));
      fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'exmp' } });
      fireEvent.change(screen.getByLabelText(/economic theme id/i), {
        target: { value: shadowCompletedJob.economic_theme_id },
      });
      fireEvent.click(screen.getByRole('button', { name: /request research/i }));
      expect(await screen.findByText(/shadow preview/i)).toBeVisible();

      // Same sealed revision, but a hold was recorded since it was loaded.
      const held = {
        ...shadowPreview,
        claims: [{ ...shadowPreview.claims[0], active_holds: ['stale'], freshness_state: 'stale' }],
      };
      api.getResearchJobPreview.mockResolvedValue(held);
      await vi.advanceTimersByTimeAsync(PREVIEW_REFRESH_MS + 1);
      expect(await screen.findByText('Hold: stale')).toBeVisible();
    } finally {
      vi.useRealTimers();
    }
  });

  it('surfaces typed request errors', async () => {
    api.requestExposureResearch.mockRejectedValue({
      response: { data: { detail: { code: 'research_disabled' } } },
    });
    renderWithProviders(<ExposureResearchWorkspace />);
    fireEvent.change(screen.getByLabelText(/admin key/i), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));
    fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'EXMP' } });
    fireEvent.change(screen.getByLabelText(/economic theme id/i), { target: { value: 't1' } });
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));
    expect(await screen.findByText('research_disabled')).toBeVisible();
    expect(api.getResearchJob).not.toHaveBeenCalled();
  });

  it('retries a failed submission with the same idempotency key', async () => {
    api.requestExposureResearch
      .mockRejectedValueOnce({ message: 'Network Error' })
      .mockResolvedValueOnce({
        job_id: shadowCompletedJob.job_id, created: false, state: 'queued', dispatch: 'queued',
      });
    api.getResearchJob.mockResolvedValue(shadowCompletedJob);
    api.getResearchJobPreview.mockResolvedValue(shadowPreview);
    renderWithProviders(<ExposureResearchWorkspace />);
    fireEvent.change(screen.getByLabelText(/admin key/i), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));
    fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'EXMP' } });
    fireEvent.change(screen.getByLabelText(/economic theme id/i), { target: { value: 't1' } });
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));
    expect(await screen.findByText('Network Error')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));

    await waitFor(() => expect(api.requestExposureResearch).toHaveBeenCalledTimes(2));
    const [first, second] = api.requestExposureResearch.mock.calls.map(([, body]) => body);
    expect(second.idempotencyKey).toBe(first.idempotencyKey);
  });

  it('uses a new idempotency key when the request changes', async () => {
    api.requestExposureResearch.mockRejectedValue({ message: 'Network Error' });
    renderWithProviders(<ExposureResearchWorkspace />);
    fireEvent.change(screen.getByLabelText(/admin key/i), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));
    fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'EXMP' } });
    fireEvent.change(screen.getByLabelText(/economic theme id/i), { target: { value: 't1' } });
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));
    expect(await screen.findByText('Network Error')).toBeVisible();
    fireEvent.change(screen.getByLabelText(/us symbol/i), { target: { value: 'OTHR' } });
    fireEvent.click(screen.getByRole('button', { name: /request research/i }));

    await waitFor(() => expect(api.requestExposureResearch).toHaveBeenCalledTimes(2));
    const [first, second] = api.requestExposureResearch.mock.calls.map(([, body]) => body);
    expect(second.idempotencyKey).not.toBe(first.idempotencyKey);
  });
});
