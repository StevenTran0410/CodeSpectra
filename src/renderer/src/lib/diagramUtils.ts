/**
 * Diagram utility functions for edge styling and labeling based on provenance.
 *
 * Handles EXTRACTED (solid), INFERRED (dashed), and AMBIGUOUS (dotted) edge styles.
 */

import type { C4DiagramEdge, ProvenanceType } from '../types/analysis'
import type { CSSProperties } from 'react'

/**
 * Get CSS style for an edge based on its provenance type.
 *
 * - EXTRACTED (or undefined): solid line
 * - INFERRED: dashed line
 * - AMBIGUOUS: dotted line
 */
export function getEdgeStyle(provenance?: ProvenanceType): CSSProperties {
  return {
    stroke: provenance === 'AMBIGUOUS' ? '#71717a' : '#52525b',
    strokeDasharray: provenance === 'INFERRED' ? '5 3' : undefined,
    strokeWidth: 1,
  }
}

/**
 * Get display label for an edge, optionally appending provenance and confidence.
 *
 * - EXTRACTED or absent: return label as-is
 * - INFERRED: append [INFERRED] and optional confidence percentage
 * - AMBIGUOUS: append [AMBIGUOUS] and optional confidence percentage
 */
export function getEdgeLabel(edge: C4DiagramEdge): string | undefined {
  if (!edge.provenance || edge.provenance === 'EXTRACTED') {
    return edge.label
  }

  const conf = edge.confidence != null ? ` ${Math.round(edge.confidence * 100)}%` : ''
  const suffix = `[${edge.provenance}${conf}]`
  return edge.label ? `${edge.label} ${suffix}` : suffix
}
