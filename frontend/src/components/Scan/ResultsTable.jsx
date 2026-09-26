import { useMemo, useRef, useState, useCallback, memo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  Chip,
  Typography,
  Box,
  CircularProgress,
  IconButton,
  styled,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import RSSparkline from './RSSparkline';
import PriceSparkline from './PriceSparkline';
import FieldAvailabilityChip from './FieldAvailabilityChip';
import MarketBadge from './MarketBadge';
import MarketThemesList from '../Stock/MarketThemesList';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import OpportunityEvidenceDrawer from '../shared/OpportunityEvidenceDrawer';
import OpportunityResultCells from './OpportunityResultCells';
import { useOpportunityEvidenceSelection } from './useOpportunityEvidenceSelection';
import {
  getStageColor,
  getRatingColor,
  getGrowthColor,
  getEpsRatingColor,
  getGroupRankColor,
} from '../../utils/colorUtils';
import {
  formatLargeNumber,
  formatIpoAge,
  getIpoAgeColor,
  getCurrencyPrefix,
  formatLocalCurrency,
} from '../../utils/formatUtils';

// Row height constant for virtualization
const ROW_HEIGHT = 48;
const SYMBOL_COLUMN_WIDTH = 210;
const OPPORTUNITY_COLUMNS_WIDTH = 153;
const TABLE_MIN_WIDTH = Object.freeze({
  WITH_ACTIONS: 2673,
  WITHOUT_ACTIONS: 2613,
});

// MCap column display modes. USD is the default per 3axp: cross-market
// parity is the common case; local is one click away. Kept as constants
// (rather than bare strings) so callers grep-reliably and typos fail fast.
const MCAP_DISPLAY = Object.freeze({
  USD: 'usd',
  LOCAL: 'local',
});

const formatGroupRankDate = (value) => {
  if (!value) return null;
  const [year, month, day] = String(value).split('-').map((part) => Number(part));
  if (year && month && day) {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'UTC',
    }).format(new Date(Date.UTC(year, month - 1, day)));
  }
  return String(value);
};

const getGroupRankTooltip = (row) => {
  if (row.ibd_group_rank == null || !row.ibd_group_rank_date) return undefined;
  return `Group rank ${row.ibd_group_rank} as of ${formatGroupRankDate(row.ibd_group_rank_date)}`;
};

// Column definitions with explicit widths
const columns = [
  { id: 'chart', label: '', sortable: false, width: 60 },
  // Width fits "0700.HK" + MarketBadge + FieldAvailabilityChip on a single
  // line without overflow (nowrap guards the rest).
  { id: 'symbol', label: 'Sym', sortable: true, width: SYMBOL_COLUMN_WIDTH },
  { id: 'rs_trend', label: 'RS Trend', sortable: true, width: 110 },
  { id: 'price_change_1d', label: 'Price', sortable: true, width: 110 },
  { id: 'gics_sector', label: 'Sector', sortable: true, width: 80 },
  { id: 'ibd_industry_group', label: 'IBD Industry', sortable: true, width: 140 },
  { id: 'market_themes', label: 'Themes', sortable: false, width: 180 },
  { id: 'ibd_group_rank', label: 'Grp', sortable: true, width: 45 },
  { id: 'composite_score', label: 'Comp', sortable: true, width: 50 },
  { id: 'minervini_score', label: 'Min', sortable: true, width: 45 },
  { id: 'canslim_score', label: 'CAN', sortable: true, width: 45 },
  { id: 'ipo_score', label: 'IPO', sortable: true, width: 45 },
  { id: 'custom_score', label: 'Cust', sortable: true, width: 45 },
  { id: 'volume_breakthrough_score', label: 'VolB', sortable: true, width: 50 },
  { id: 'se_setup_score', label: 'SE', sortable: true, width: 45 },
  { id: 'se_pattern_primary', label: 'Pat', sortable: true, width: 55 },
  { id: 'se_distance_to_pivot_pct', label: 'Pvt%', sortable: true, width: 50 },
  { id: 'se_bb_width_pctile_252', label: 'Sqz', sortable: true, width: 45 },
  { id: 'se_volume_vs_50d', label: 'V50', sortable: true, width: 45 },
  { id: 'se_rs_line_new_high', label: 'RSH', sortable: false, width: 35 },
  { id: 'se_rs_line_blue_dot', label: 'SEBD', sortable: false, width: 42 },
  { id: 'rs_line_blue_dot_recent', label: 'RSBD', sortable: true, width: 42 },
  { id: 'se_pivot_price', label: 'Pvt$', sortable: true, width: 55 },
  { id: 'resilience_score', label: 'Res', sortable: true, width: 48 },
  { id: 'action_state', label: 'Action', sortable: true, width: 105 },
  { id: 'rs_rating', label: 'RS', sortable: true, width: 40 },
  { id: 'rs_rating_1m', label: '1M', sortable: true, width: 40 },
  { id: 'rs_rating_3m', label: '3M', sortable: true, width: 40 },
  { id: 'rs_rating_12m', label: '12M', sortable: true, width: 45 },
  { id: 'beta', label: 'β', sortable: true, width: 45 },
  { id: 'beta_adj_rs', label: 'βRS', sortable: true, width: 45 },
  { id: 'eps_rating', label: 'EPS Rtg', sortable: true, width: 55 },
  { id: 'stage', label: 'Stg', sortable: true, width: 40 },
  { id: 'current_price', label: 'Price', sortable: true, width: 65 },
  { id: 'volume', label: '$Vol', sortable: true, width: 60 },
  // MCap column header label is overridden per-render based on the USD/Local
  // toggle; keep the underlying sort key stable at 'market_cap' so the
  // sort-by dropdown / URL state doesn't shift when the user flips modes.
  { id: 'market_cap', label: 'MCap', sortable: true, width: 75 },
  { id: 'adv_usd', label: 'ADV ($)', sortable: true, width: 70 },
  { id: 'ipo_date', label: 'IPO', sortable: true, width: 50 },
  { id: 'eps_growth_qq', label: 'EPS', sortable: true, width: 50 },
  { id: 'sales_growth_qq', label: 'Sales', sortable: true, width: 50 },
  { id: 'adr_percent', label: 'ADR', sortable: true, width: 50 },
  { id: 'ma_alignment', label: 'MA', sortable: false, width: 35 },
  { id: 'vcp_detected', label: 'VCP', sortable: false, width: 40 },
  { id: 'vcp_score', label: 'VScr', sortable: true, width: 50 },
  { id: 'vcp_pivot', label: 'Pvt', sortable: true, width: 55 },
  { id: 'vcp_ready_for_breakout', label: 'Rdy', sortable: false, width: 35 },
  { id: 'passes_template', label: 'Pass', sortable: false, width: 40 },
  { id: 'rating', label: 'Rate', sortable: false, width: 80 },
];

const getStatusChipProps = (row) => {
  const isInsufficientHistoryRow =
    row.data_status === 'insufficient_history' || row.rating === 'Insufficient Data';

  if (row.scan_mode === 'listing_only' && isInsufficientHistoryRow) {
    return {
      label: 'New IPO',
      color: 'warning',
      title: 'Visible in the scan table, but not yet scannable because price history is still limited.',
    };
  }
  if (row.scan_mode === 'ipo_weighted' && isInsufficientHistoryRow) {
    return {
      label: 'IPO Weighted',
      color: 'info',
      title: row.composite_reason === 'ipo_uplift'
        ? 'Composite uses applicable screeners plus an IPO uplift while the stock is still young.'
        : 'Composite uses only the screeners that have enough history to run.',
    };
  }
  return null;
};

// Static body-cell styles live on the row as class selectors so each render
// only resolves one style object per row, instead of running MUI's `sx`
// resolution for ~48 cells per row on every page, sort, or scroll update.
const CELL_WIDTHS = [35, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 110, 140, 180];
const COLOR_CLASS_PATHS = [
  'text.primary',
  'text.secondary',
  'text.disabled',
  'primary.main',
  'success.main',
  'success.light',
  'warning.main',
  'error.main',
];
const colorClass = (path) => (
  COLOR_CLASS_PATHS.includes(path) ? `rt-c-${path.replace('.', '-')}` : ''
);
const resolvePaletteColor = (palette, path) => (
  path.split('.').reduce((value, key) => value?.[key], palette)
);
const stopPropagation = (event) => event.stopPropagation();

const ResultRow = styled(TableRow)(({ theme }) => ({
  height: ROW_HEIGHT,
  cursor: 'default',
  '&.rt-clickable': { cursor: 'pointer' },
  ...Object.fromEntries(CELL_WIDTHS.map((width) => [
    `& .rt-w${width}`,
    { width, minWidth: width },
  ])),
  ...Object.fromEntries(COLOR_CLASS_PATHS.map((path) => [
    `& .${colorClass(path)}`,
    { color: resolvePaletteColor(theme.palette, path) },
  ])),
  '& .rt-mono': { fontFamily: 'monospace' },
  '& .rt-strong': { fontWeight: 600 },
  '& .rt-ellipsis': { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  '& .rt-p2': { padding: '2px' },
  '& .rt-p4': { padding: '4px' },
  '& .rt-py4': { paddingTop: theme.spacing(0.5), paddingBottom: theme.spacing(0.5) },
  '& .rt-symbol': {
    width: SYMBOL_COLUMN_WIDTH,
    minWidth: SYMBOL_COLUMN_WIDTH,
    maxWidth: SYMBOL_COLUMN_WIDTH,
    paddingTop: '4px',
    paddingBottom: '4px',
    overflow: 'hidden',
  },
  '& .rt-sym-stack': {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(0.25),
    minWidth: 0,
  },
  '& .rt-sym-line': {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(0.5),
    minWidth: 0,
    whiteSpace: 'nowrap',
  },
  '& .rt-sym-sub': {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(0.5),
    minWidth: 0,
  },
  '& .rt-sym-text': { fontWeight: 600, lineHeight: 1.2, flexShrink: 0 },
  '& .rt-sym-company': { display: 'block', lineHeight: 1.2, minWidth: 0, flex: 1 },
  '& .rt-chart-button': { color: theme.palette.primary.main, padding: 0 },
  '& .rt-icon': { fontSize: 14 },
  '& .rt-dot': {
    width: 10,
    height: 10,
    borderRadius: '50%',
    backgroundColor: '#2196f3',
    display: 'inline-block',
  },
  '& .rt-stage': {
    color: 'white',
    padding: '1px 4px',
    borderRadius: '2px',
    fontSize: '10px',
    fontWeight: 500,
  },
}));

/**
 * Memoized table row component to prevent unnecessary re-renders
 */
const VirtualTableRow = memo(function VirtualTableRow({
  row,
  onRowClick,
  onRowHover,
  onOpenChart,
  showActions,
  showWatchlistMenu,
  chartEnabled,
  mcapDisplay,
  showOpportunityState,
  onOpenOpportunity,
}) {
  const statusChip = getStatusChipProps(row);
  const handleRowClick = useCallback(() => {
    if (!chartEnabled) {
      return;
    }
    onRowClick?.(row.symbol);
  }, [chartEnabled, onRowClick, row.symbol]);

  const handleRowHover = useCallback(() => {
    onRowHover?.(row.symbol);
  }, [onRowHover, row.symbol]);

  const handleChartClick = useCallback((e) => {
    e.stopPropagation();
    if (!chartEnabled) {
      return;
    }
    onOpenChart?.(row.symbol);
  }, [chartEnabled, onOpenChart, row.symbol]);

  const check = (value, falseClass = 'rt-c-text-disabled') => (
    value
      ? <CheckIcon className="rt-icon rt-c-success-main" />
      : <CloseIcon className={`rt-icon ${falseClass}`} />
  );

  return (
    <ResultRow
      hover
      onClick={handleRowClick}
      onMouseEnter={handleRowHover}
      className={onRowClick && chartEnabled ? 'rt-clickable' : undefined}
    >
      {showActions && (
        <TableCell align="center" onClick={stopPropagation} className="rt-p2 rt-w60">
          {chartEnabled ? (
            <IconButton
              size="small"
              onClick={handleChartClick}
              className="rt-chart-button"
            >
              <ShowChartIcon className="rt-icon" />
            </IconButton>
          ) : null}
          {showWatchlistMenu ? <AddToWatchlistMenu symbols={row.symbol} size="small" /> : null}
        </TableCell>
      )}

      <TableCell className="rt-symbol">
        <div className="rt-sym-stack">
          <div className="rt-sym-line">
            <Typography component="span" variant="body2" className="rt-sym-text">
              {row.symbol}
            </Typography>
            <MarketBadge market={row.market} exchange={row.exchange} />
            <FieldAvailabilityChip
              fieldAvailability={row.field_availability}
              growthMetricBasis={row.growth_metric_basis}
            />
          </div>
          {row.company_name || statusChip || row.matched_groups?.length ? (
            <div className="rt-sym-sub">
              {row.company_name ? (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  noWrap
                  title={row.company_name}
                  className="rt-sym-company"
                >
                  {row.company_name}
                </Typography>
              ) : null}
              {statusChip ? (
                <Chip
                  label={statusChip.label}
                  color={statusChip.color}
                  size="small"
                  title={statusChip.title}
                  sx={{ height: 18, fontSize: 10, flexShrink: 0 }}
                />
              ) : null}
              {row.matched_groups?.length ? (
                <Chip
                  label={row.matched_groups.length === 1
                    ? row.matched_groups[0].name
                    : `${row.matched_groups[0].name} +${row.matched_groups.length - 1}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                  title={`Matched setups: ${row.matched_groups.map((group) => group.name).join(', ')}`}
                  sx={{
                    height: 18,
                    maxWidth: 105,
                    fontSize: 9,
                    flexShrink: 0,
                    '& .MuiChip-label': { px: 0.65, overflow: 'hidden', textOverflow: 'ellipsis' },
                  }}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </TableCell>

      <TableCell align="center" className="rt-p4 rt-w110">
        <RSSparkline
          data={row.rs_sparkline_data}
          trend={row.rs_trend}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" className="rt-p4 rt-w110">
        <PriceSparkline
          data={row.price_sparkline_data}
          trend={row.price_trend}
          change1d={row.price_change_1d}
          industry={row.ibd_industry_group}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" className="rt-c-text-secondary rt-w80 rt-ellipsis">
        {row.gics_sector || '-'}
      </TableCell>

      <TableCell align="left" className="rt-c-text-secondary rt-w140 rt-ellipsis">
        {row.ibd_industry_group || '-'}
      </TableCell>

      <TableCell align="left" className="rt-c-text-secondary rt-w180 rt-py4">
        <MarketThemesList themes={row.market_themes} variant="compact" />
      </TableCell>

      <TableCell
        align="center"
        title={getGroupRankTooltip(row)}
        className={`rt-mono rt-w45 ${colorClass(getGroupRankColor(row.ibd_group_rank))}${
          row.ibd_group_rank && row.ibd_group_rank <= 20 ? ' rt-strong' : ''
        }`}
      >
        {row.ibd_group_rank ?? '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50 rt-strong rt-c-primary-main">
        {row.composite_score?.toFixed(1) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.minervini_score != null ? row.minervini_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.canslim_score != null ? row.canslim_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.ipo_score != null ? row.ipo_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.custom_score != null ? row.custom_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.volume_breakthrough_score != null ? row.volume_breakthrough_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.se_setup_score != null ? row.se_setup_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-c-text-secondary rt-w55 rt-ellipsis">
        {row.se_pattern_primary || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.se_distance_to_pivot_pct != null ? `${row.se_distance_to_pivot_pct.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.se_bb_width_pctile_252 != null ? row.se_bb_width_pctile_252.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.se_volume_vs_50d != null ? `${row.se_volume_vs_50d.toFixed(1)}x` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {row.se_rs_line_new_high == null ? '-' : check(row.se_rs_line_new_high)}
      </TableCell>

      <TableCell align="center" className="rt-w42">
        {row.se_rs_line_blue_dot ? (
          <span className="rt-dot" title="Setup Engine RS line at new high before price" />
        ) : (
          <span className="rt-c-text-disabled">-</span>
        )}
      </TableCell>

      <TableCell align="center" className="rt-w42">
        {row.rs_line_blue_dot_recent ? (
          <span
            className="rt-dot"
            title={row.rs_line_new_high_date ? `RS blue dot: latest RS high ${row.rs_line_new_high_date}` : 'RS blue dot within 5 trading days'}
          />
        ) : (
          <span className="rt-c-text-disabled">-</span>
        )}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w55">
        {row.se_pivot_price != null ? `$${row.se_pivot_price.toFixed(2)}` : '-'}
      </TableCell>

      <OpportunityResultCells
        row={row}
        visible={showOpportunityState}
        onOpenEvidence={onOpenOpportunity}
      />

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating_1m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating_3m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.rs_rating_12m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.beta != null ? row.beta.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.beta_adj_rs != null ? row.beta_adj_rs.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w55 ${colorClass(getEpsRatingColor(row.eps_rating))}`}>
        {row.eps_rating != null ? row.eps_rating : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {row.stage != null ? (
          <span className="rt-stage" style={{ backgroundColor: getStageColor(row.stage) }}>
            S{row.stage}
          </span>
        ) : (
          '-'
        )}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w65">
        {formatLocalCurrency(row.current_price, row.currency)}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w60">
        {formatLargeNumber(row.volume, getCurrencyPrefix(row.currency))}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w75">
        {mcapDisplay === MCAP_DISPLAY.USD
          ? formatLargeNumber(row.market_cap_usd, '$')
          : formatLargeNumber(row.market_cap, getCurrencyPrefix(row.currency))}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w70">
        {formatLargeNumber(row.adv_usd, '$')}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getIpoAgeColor(row.ipo_date))}`}>
        {formatIpoAge(row.ipo_date)}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getGrowthColor(row.eps_growth_qq))}`}>
        {row.eps_growth_qq != null ? `${row.eps_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getGrowthColor(row.sales_growth_qq))}`}>
        {row.sales_growth_qq != null ? `${row.sales_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.adr_percent != null ? `${row.adr_percent.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {check(row.ma_alignment, 'rt-c-error-main')}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {check(row.vcp_detected)}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.vcp_score != null ? row.vcp_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w55">
        {row.vcp_pivot != null ? row.vcp_pivot.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {check(row.vcp_ready_for_breakout)}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {check(row.passes_template)}
      </TableCell>

      <TableCell align="center" className="rt-w80">
        <Chip
          label={row.rating}
          color={getRatingColor(row.rating)}
          size="small"
        />
      </TableCell>
    </ResultRow>
  );
});

/**
 * Display scan results in a sortable, paginated table with row virtualization
 * @param {Function} onRowHover - Optional callback when hovering over a row (for prefetching)
 */
function ResultsTable({
  results,
  total,
  page,
  perPage,
  sortBy,
  sortOrder,
  onPageChange,
  onPerPageChange,
  onSortChange,
  onOpenChart,
  loading,
  onRowHover,
  showActions = true,
  showWatchlistMenu = true,
  sortingEnabled = true,
  isChartEnabled,
  showOpportunityState = false,
  opportunityTelemetrySurface,
}) {
  const parentRef = useRef(null);
  // MCap column display mode — kept as local state; scan-level persistence
  // can lift this up later if users want it to survive navigation.
  const [mcapDisplay, setMcapDisplay] = useState(MCAP_DISPLAY.USD);
  const {
    selectedRow: opportunityRow,
    openEvidence: handleOpenOpportunity,
    closeEvidence: handleCloseOpportunity,
  } = useOpportunityEvidenceSelection({
    rows: results,
    enabled: showOpportunityState,
  });
  const visibleColumns = useMemo(() => {
    const base = columns.filter((column) => (
      (showActions || column.id !== 'chart') &&
      (showOpportunityState || !['resilience_score', 'action_state'].includes(column.id))
    ));
    return base.map((column) =>
      column.id === 'market_cap'
        ? { ...column, label: mcapDisplay === MCAP_DISPLAY.USD ? 'MCap ($)' : 'MCap (local)' }
        : column,
    );
  }, [showActions, showOpportunityState, mcapDisplay]);

  const toggleMcapDisplay = useCallback(() => {
    setMcapDisplay((mode) =>
      mode === MCAP_DISPLAY.USD ? MCAP_DISPLAY.LOCAL : MCAP_DISPLAY.USD,
    );
  }, []);

  const handleChangePage = useCallback((event, newPage) => {
    onPageChange(newPage + 1); // Material-UI uses 0-based pages, API uses 1-based
  }, [onPageChange]);

  const handleRequestSort = useCallback((property) => {
    const isAsc = sortBy === property && sortOrder === 'asc';
    onSortChange(property, isAsc ? 'desc' : 'asc');
  }, [sortBy, sortOrder, onSortChange]);

  const handleRowClick = useCallback((symbol) => {
    onOpenChart?.(symbol);
  }, [onOpenChart]);

  // Virtualize rows - only render visible rows plus overscan
  const rowVirtualizer = useVirtualizer({
    count: results?.length || 0,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10, // Render 10 extra rows above/below viewport
  });

  // Memoize virtual items to prevent recalculation
  const virtualRows = rowVirtualizer.getVirtualItems();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!results || results.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No results found
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={1}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', px: 2, py: 0.5, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
          Market Cap display:
        </Typography>
        <Chip
          label={mcapDisplay === MCAP_DISPLAY.USD ? 'USD' : 'Local'}
          size="small"
          variant="outlined"
          onClick={toggleMcapDisplay}
          data-testid="mcap-display-toggle"
          sx={{ cursor: 'pointer', fontSize: 11, height: 20 }}
        />
      </Box>
      <TableContainer
        ref={parentRef}
        sx={{
          maxHeight: 'calc(100vh - 280px)',
          overflow: 'auto',
        }}
      >
        <Table
          stickyHeader
          size="small"
          sx={{
            minWidth: (showActions
              ? TABLE_MIN_WIDTH.WITH_ACTIONS
              : TABLE_MIN_WIDTH.WITHOUT_ACTIONS) +
              (showOpportunityState ? OPPORTUNITY_COLUMNS_WIDTH : 0),
          }}
        >
          <TableHead>
            <TableRow>
              {visibleColumns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.id === 'symbol' ? 'left' : 'center'}
                  sx={{
                    width: column.width,
                    minWidth: column.width,
                    maxWidth: column.width,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {column.sortable && sortingEnabled ? (
                    <TableSortLabel
                      active={sortBy === column.id}
                      direction={sortBy === column.id ? sortOrder : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {/* Spacer for virtualization - pushes content down to correct position */}
            {virtualRows.length > 0 && virtualRows[0].start > 0 && (
              <tr style={{ height: virtualRows[0].start }} />
            )}
            {virtualRows.map((virtualRow) => {
              const row = results[virtualRow.index];
              return (
                <VirtualTableRow
                  key={row.symbol}
                  row={row}
                  onRowClick={onOpenChart ? handleRowClick : null}
                  onRowHover={onRowHover}
                  onOpenChart={onOpenChart}
                  showActions={showActions}
                  showWatchlistMenu={showWatchlistMenu}
                  chartEnabled={
                    row.is_scannable !== false &&
                    (isChartEnabled ? isChartEnabled(row.symbol) : Boolean(onOpenChart))
                  }
                  mcapDisplay={mcapDisplay}
                  showOpportunityState={showOpportunityState}
                  onOpenOpportunity={handleOpenOpportunity}
                />
              );
            })}
            {/* Bottom spacer for virtualization */}
            {virtualRows.length > 0 && (
              <tr style={{ height: rowVirtualizer.getTotalSize() - (virtualRows[virtualRows.length - 1]?.end || 0) }} />
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={total}
        rowsPerPage={perPage}
        page={page - 1} // Material-UI uses 0-based pages
        onPageChange={handleChangePage}
        onRowsPerPageChange={(e) => {
          const nextPerPage = Number(e.target.value);
          onPerPageChange?.(nextPerPage);
          onPageChange(1); // Reset to first page when changing per-page
        }}
      />

      <OpportunityEvidenceDrawer
        open={showOpportunityState && Boolean(opportunityRow)}
        row={opportunityRow}
        onClose={handleCloseOpportunity}
        opportunityTelemetrySurface={opportunityTelemetrySurface}
      />
    </Paper>
  );
}

// Wrap with React.memo for component-level memoization
export default memo(ResultsTable, (prevProps, nextProps) => {
  // Only re-render if these key props change
  return (
    prevProps.results === nextProps.results &&
    prevProps.total === nextProps.total &&
    prevProps.page === nextProps.page &&
    prevProps.perPage === nextProps.perPage &&
    prevProps.sortBy === nextProps.sortBy &&
    prevProps.sortOrder === nextProps.sortOrder &&
    prevProps.loading === nextProps.loading &&
    prevProps.showActions === nextProps.showActions &&
    prevProps.showWatchlistMenu === nextProps.showWatchlistMenu &&
    prevProps.isChartEnabled === nextProps.isChartEnabled &&
    prevProps.sortingEnabled === nextProps.sortingEnabled &&
    prevProps.showOpportunityState === nextProps.showOpportunityState &&
    prevProps.opportunityTelemetrySurface === nextProps.opportunityTelemetrySurface
  );
});
