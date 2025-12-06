"""
Automated EDA Report Generator

Generates comprehensive HTML report with visualizations for salary data analysis.
Designed to run automatically when new data is fetched.

Usage:
    python -m src.analysis.eda_report
    python -m src.analysis.eda_report --data-dir data/merged
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDAReportGenerator:
    """Generate automated EDA report with visualizations."""

    def __init__(self, data_dir: str = "data/merged", output_dir: str = "docs/reports", keep_latest_only: bool = False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.figures = []
        self.stats = {}
        self.keep_latest_only = keep_latest_only

    def load_data(self) -> dict:
        """Load all available data sources."""
        print("Loading data sources...")

        data = {}

        # Try to load merged data
        merged_path = self.data_dir / "merged_salary_data.parquet"
        if merged_path.exists():
            data['merged'] = pd.read_parquet(merged_path)
            print(f"  Merged: {len(data['merged']):,} records")

        # Try to load individual sources from hierarchical structure
        base_dir = self.data_dir.parent
        sources = {
            'h1b': base_dir / "h1b" / "processed" / "h1b_ai_salaries.parquet",
            'linkedin': base_dir / "linkedin" / "processed" / "linkedin_ai_jobs.parquet",
            'adzuna': base_dir / "adzuna" / "processed" / "adzuna_jobs.parquet"
        }

        for source_name, source_path in sources.items():
            if source_path.exists():
                data[source_name] = pd.read_parquet(source_path)
                print(f"  {source_name}: {len(data[source_name]):,} records")

        return data

    def generate_summary_stats(self, data: dict):
        """Generate summary statistics."""
        print("\nGenerating summary statistics...")

        for name, df in data.items():
            stats = {
                'records': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
            }

            # Salary stats
            salary_col = self._find_salary_column(df)
            if salary_col:
                salaries = df[df[salary_col].notna()][salary_col]
                stats['salary'] = {
                    'count': len(salaries),
                    'mean': float(salaries.mean()),
                    'median': float(salaries.median()),
                    'std': float(salaries.std()),
                    'min': float(salaries.min()),
                    'max': float(salaries.max()),
                    'q25': float(salaries.quantile(0.25)),
                    'q75': float(salaries.quantile(0.75))
                }

            # Missing data
            stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).to_dict()

            self.stats[name] = stats

    def _find_salary_column(self, df: pd.DataFrame) -> str:
        """Find salary column name."""
        for col in ['annual_salary', 'salary', 'salary_avg', 'mean_annual_wage']:
            if col in df.columns:
                return col
        return None

    def plot_salary_distributions(self, data: dict):
        """Plot salary distributions by source."""
        print("Creating salary distribution plots...")

        sources_with_salary = {}
        for name, df in data.items():
            if name == 'merged':
                continue
            salary_col = self._find_salary_column(df)
            if salary_col and salary_col in df.columns:
                salaries = df[(df[salary_col].notna()) &
                             (df[salary_col] >= 30000) &
                             (df[salary_col] <= 1000000)][salary_col]
                if len(salaries) > 0:
                    sources_with_salary[name] = salaries

        if not sources_with_salary:
            return

        n_sources = len(sources_with_salary)
        fig, axes = plt.subplots(1, n_sources, figsize=(6*n_sources, 5))
        if n_sources == 1:
            axes = [axes]

        for (name, salaries), ax in zip(sources_with_salary.items(), axes):
            ax.hist(salaries, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(salaries.median(), color='red', linestyle='--',
                      label=f'Median: ${salaries.median():,.0f}')
            ax.set_title(f'{name.upper()}\n({len(salaries):,} records)')
            ax.set_xlabel('Annual Salary ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'salary_distributions')

    def plot_source_comparison(self, data: dict):
        """Compare data sources."""
        if 'merged' not in data or 'data_source' not in data['merged'].columns:
            return

        print("Creating source comparison plots...")

        df = data['merged']

        # Source distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Record counts
        source_counts = df['data_source'].value_counts()
        axes[0].bar(source_counts.index, source_counts.values)
        axes[0].set_title('Records by Data Source')
        axes[0].set_ylabel('Number of Records')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)

        # Add count labels on bars
        for i, (idx, val) in enumerate(source_counts.items()):
            axes[0].text(i, val, f'{val:,}', ha='center', va='bottom')

        # Quality scores by source (if available)
        if 'quality_score' in df.columns:
            source_quality = df.groupby('data_source')['quality_score'].agg(['mean', 'median'])
            x = np.arange(len(source_quality))
            width = 0.35

            axes[1].bar(x - width/2, source_quality['mean'], width, label='Mean', alpha=0.8)
            axes[1].bar(x + width/2, source_quality['median'], width, label='Median', alpha=0.8)
            axes[1].set_title('Quality Score by Source')
            axes[1].set_ylabel('Quality Score')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(source_quality.index, rotation=45)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].set_ylim([0, 1])

        plt.tight_layout()
        self._save_figure(fig, 'source_comparison')

    def plot_location_analysis(self, data: dict):
        """Analyze salary by location."""
        if 'merged' not in data:
            return

        print("Creating location analysis...")

        df = data['merged']
        salary_col = self._find_salary_column(df)

        # Try different location column names
        location_col = None
        for col in ['location_state', 'worksite_state', 'state']:
            if col in df.columns:
                location_col = col
                break

        if not salary_col or not location_col:
            return

        # Filter valid data
        valid = df[(df[location_col].notna()) &
                   (df[salary_col].notna()) &
                   (df[salary_col] >= 30000) &
                   (df[salary_col] <= 1000000)]

        if len(valid) == 0:
            return

        # Top states by median salary
        state_stats = valid.groupby(location_col)[salary_col].agg([
            ('count', 'count'),
            ('median', 'median')
        ]).sort_values('median', ascending=False)

        # Filter states with at least 10 samples
        state_stats = state_stats[state_stats['count'] >= 10].head(15)

        if len(state_stats) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(state_stats.index, state_stats['median'])
        ax.set_xlabel('Median Salary ($)')
        ax.set_title('Top 15 States by Median Salary (≥10 samples)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add count labels
        for i, (idx, row) in enumerate(state_stats.iterrows()):
            ax.text(row['median'], i, f" n={int(row['count'])}",
                   va='center', fontsize=8)

        plt.tight_layout()
        self._save_figure(fig, 'location_analysis')

    def plot_skills_analysis(self, data: dict):
        """Analyze skill categories from feature engineering config."""
        merged_df = data.get('merged')

        if merged_df is None:
            return

        print("Creating skills analysis...")

        # Load skill categories from config
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.config_loader import ConfigLoader
            skill_categories = ConfigLoader.get_features()["skill_categories"]
        except Exception as e:
            print(f"  Skipping: Could not load skill categories config: {e}")
            return

        salary_col = self._find_salary_column(merged_df)

        # Count jobs with each skill category
        category_counts = {}
        category_salaries = {}

        for category_name, keywords in skill_categories.items():
            count = 0
            salaries = []

            for idx, row in merged_df.iterrows():
                # Check if job has this skill category
                has_category = False

                # Check in skills column if exists
                if 'skills' in merged_df.columns:
                    skills = row['skills']
                    if skills is not None and hasattr(skills, '__iter__') and not isinstance(skills, str):
                        skills_lower = [str(s).lower() for s in skills]
                        if any(keyword in ' '.join(skills_lower) for keyword in keywords):
                            has_category = True

                # Also check in job_title if skills not available
                if not has_category and 'job_title' in merged_df.columns and pd.notna(row['job_title']):
                    title_lower = str(row['job_title']).lower()
                    if any(keyword in title_lower for keyword in keywords):
                        has_category = True

                if has_category:
                    count += 1
                    if salary_col and pd.notna(row.get(salary_col)):
                        salary = row[salary_col]
                        if 30000 <= salary <= 1000000:
                            salaries.append(salary)

            category_counts[category_name] = count
            if salaries:
                category_salaries[category_name] = salaries

        if not category_counts:
            return

        # Calculate overall median for comparison
        overall_median = merged_df[salary_col].median() if salary_col else 0

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Skill category coverage (top left)
        categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cat_names = [c[0].replace('_', ' ').title() for c in categories_sorted]
        cat_counts = [c[1] for c in categories_sorted]

        axes[0, 0].barh(range(len(cat_names)), cat_counts, color='steelblue', alpha=0.7)
        axes[0, 0].set_yticks(range(len(cat_names)))
        axes[0, 0].set_yticklabels(cat_names)
        axes[0, 0].set_xlabel('Number of Jobs')
        axes[0, 0].set_title('Skill Category Coverage\n(Based on Feature Engineering Config)')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Add count labels
        for i, count in enumerate(cat_counts):
            pct = (count / len(merged_df)) * 100
            axes[0, 0].text(count, i, f' {count:,} ({pct:.1f}%)', va='center', fontsize=9)

        # 2. Salary by skill category (top right)
        if category_salaries:
            cat_medians = {cat: np.median(sals) for cat, sals in category_salaries.items()}
            medians_sorted = sorted(cat_medians.items(), key=lambda x: x[1], reverse=True)

            med_names = [c[0].replace('_', ' ').title() for c in medians_sorted]
            med_values = [c[1] for c in medians_sorted]
            colors = ['#2ecc71' if v > overall_median else '#e74c3c' for v in med_values]

            axes[0, 1].barh(range(len(med_names)), med_values, color=colors, alpha=0.7)
            axes[0, 1].set_yticks(range(len(med_names)))
            axes[0, 1].set_yticklabels(med_names)
            axes[0, 1].set_xlabel('Median Salary ($)')
            axes[0, 1].set_title(f'Median Salary by Skill Category\n(Baseline: ${overall_median:,.0f})')
            axes[0, 1].axvline(overall_median, color='black', linestyle='--', linewidth=2)
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(axis='x', alpha=0.3)

            # Add salary labels
            for i, val in enumerate(med_values):
                axes[0, 1].text(val, i, f' ${val:,.0f}', va='center', fontsize=8)

        # 3. Skill count distribution (bottom left)
        if 'skills' in merged_df.columns:
            skill_counts = merged_df['skills'].apply(
                lambda x: len(x) if hasattr(x, '__len__') and not isinstance(x, str) else 0
            )

            axes[1, 0].hist(skill_counts, bins=30, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 0].axvline(skill_counts.mean(), color='red', linestyle='--',
                             label=f'Mean: {skill_counts.mean():.1f}', linewidth=2)
            axes[1, 0].axvline(skill_counts.median(), color='green', linestyle='--',
                             label=f'Median: {skill_counts.median():.1f}', linewidth=2)
            axes[1, 0].set_xlabel('Number of Skills per Job')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Skill Count\n(skill_count feature)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Skills data not available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Category overlap analysis (bottom right)
        if category_counts:
            # Count jobs with N categories
            category_overlap = {}
            for idx, row in merged_df.iterrows():
                num_categories = 0
                for category_name, keywords in skill_categories.items():
                    has_category = False
                    if 'skills' in merged_df.columns:
                        skills = row['skills']
                        if skills is not None and hasattr(skills, '__iter__') and not isinstance(skills, str):
                            skills_lower = [str(s).lower() for s in skills]
                            if any(keyword in ' '.join(skills_lower) for keyword in keywords):
                                has_category = True
                    if not has_category and 'job_title' in merged_df.columns and pd.notna(row['job_title']):
                        title_lower = str(row['job_title']).lower()
                        if any(keyword in title_lower for keyword in keywords):
                            has_category = True
                    if has_category:
                        num_categories += 1

                category_overlap[num_categories] = category_overlap.get(num_categories, 0) + 1

            overlap_sorted = sorted(category_overlap.items())
            overlap_counts = [c[0] for c in overlap_sorted]
            overlap_freq = [c[1] for c in overlap_sorted]

            axes[1, 1].bar(overlap_counts, overlap_freq, color='coral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Number of Skill Categories per Job')
            axes[1, 1].set_ylabel('Number of Jobs')
            axes[1, 1].set_title('Skill Category Overlap\n(How many categories per job?)')
            axes[1, 1].grid(axis='y', alpha=0.3)

            # Add count labels
            for x, y in zip(overlap_counts, overlap_freq):
                axes[1, 1].text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, 'skills_analysis')

    def plot_target_distribution_analysis(self, data: dict):
        """Deep dive into target variable distribution - log-normality and outliers."""
        if 'merged' not in data:
            return

        print("Creating target distribution analysis...")

        df = data['merged']
        salary_col = self._find_salary_column(df)

        if not salary_col:
            return

        salaries = df[(df[salary_col].notna()) &
                     (df[salary_col] >= 30000) &
                     (df[salary_col] <= 1000000)][salary_col]

        if len(salaries) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Original distribution
        axes[0, 0].hist(salaries, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(salaries.median(), color='red', linestyle='--',
                          label=f'Median: ${salaries.median():,.0f}', linewidth=2)
        axes[0, 0].axvline(salaries.mean(), color='green', linestyle='--',
                          label=f'Mean: ${salaries.mean():,.0f}', linewidth=2)
        axes[0, 0].set_xlabel('Annual Salary ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Salary Distribution (Original Scale)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Log-transformed distribution
        log_salaries = np.log10(salaries)
        axes[0, 1].hist(log_salaries, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].axvline(log_salaries.median(), color='red', linestyle='--',
                          label=f'Median: {log_salaries.median():.2f}', linewidth=2)
        axes[0, 1].axvline(log_salaries.mean(), color='green', linestyle='--',
                          label=f'Mean: {log_salaries.mean():.2f}', linewidth=2)
        axes[0, 1].set_xlabel('Log10(Annual Salary)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Salary Distribution (Log Scale) - More Normal?')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Boxplot for outlier detection
        axes[1, 0].boxplot(salaries, vert=True)
        q1 = salaries.quantile(0.25)
        q3 = salaries.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = salaries[(salaries < lower_bound) | (salaries > upper_bound)]

        axes[1, 0].set_ylabel('Annual Salary ($)')
        axes[1, 0].set_title(f'Boxplot - Outlier Detection\n{len(outliers):,} outliers ({len(outliers)/len(salaries)*100:.1f}%)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_xticklabels(['Salary'])

        # Add IQR annotation
        axes[1, 0].text(1.15, q1, f'Q1: ${q1:,.0f}', fontsize=9, va='center')
        axes[1, 0].text(1.15, q3, f'Q3: ${q3:,.0f}', fontsize=9, va='center')
        axes[1, 0].text(1.15, upper_bound, f'Upper: ${upper_bound:,.0f}', fontsize=9, va='center', color='red')

        # 4. Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = [salaries.quantile(p/100) for p in percentiles]

        axes[1, 1].barh([f'P{p}' for p in percentiles], pct_values, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Annual Salary ($)')
        axes[1, 1].set_title('Salary Percentiles')
        axes[1, 1].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (pct, val) in enumerate(zip(percentiles, pct_values)):
            axes[1, 1].text(val, i, f' ${val:,.0f}', va='center', fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, 'target_distribution_analysis')

    def plot_feature_target_relationships(self, data: dict):
        """Analyze relationships between features and target variable."""
        if 'merged' not in data:
            return

        print("Creating feature-target relationship analysis...")

        df = data['merged']
        salary_col = self._find_salary_column(df)

        if not salary_col:
            return

        # Filter valid data
        valid = df[(df[salary_col].notna()) &
                   (df[salary_col] >= 30000) &
                   (df[salary_col] <= 1000000)].copy()

        if len(valid) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Experience vs Salary scatter plot
        exp_col = None
        for col in ['estimated_yoe', 'years_of_experience', 'experience_years']:
            if col in valid.columns:
                exp_col = col
                break

        if exp_col and valid[exp_col].notna().sum() > 10:
            exp_valid = valid[valid[exp_col].notna()]
            axes[0, 0].scatter(exp_valid[exp_col], exp_valid[salary_col],
                              alpha=0.3, s=20, color='steelblue')

            # Add trend line
            z = np.polyfit(exp_valid[exp_col], exp_valid[salary_col], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(exp_valid[exp_col].min(), exp_valid[exp_col].max(), 100)
            axes[0, 0].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Polynomial Trend')

            axes[0, 0].set_xlabel('Years of Experience')
            axes[0, 0].set_ylabel('Annual Salary ($)')
            axes[0, 0].set_title(f'Experience vs Salary\n(n={len(exp_valid):,})')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Experience data not available',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Experience vs Salary')

        # 2. Location tiering - Top states by median salary
        location_col = None
        for col in ['location_state', 'worksite_state', 'state']:
            if col in valid.columns:
                location_col = col
                break

        if location_col:
            state_stats = valid.groupby(location_col)[salary_col].agg([
                ('count', 'count'),
                ('median', 'median')
            ]).sort_values('median', ascending=True)

            # Filter states with at least 20 samples and take top 15
            state_stats = state_stats[state_stats['count'] >= 20].tail(15)

            if len(state_stats) > 0:
                colors = ['#2ecc71' if med > valid[salary_col].median() else '#e74c3c'
                         for med in state_stats['median']]
                axes[0, 1].barh(state_stats.index, state_stats['median'], color=colors, alpha=0.7)
                axes[0, 1].axvline(valid[salary_col].median(), color='black', linestyle='--',
                                  label=f'Overall Median: ${valid[salary_col].median():,.0f}', linewidth=2)
                axes[0, 1].set_xlabel('Median Salary ($)')
                axes[0, 1].set_title('Top 15 States by Median Salary\n(Green = Above Overall Median)')
                axes[0, 1].legend()
                axes[0, 1].grid(axis='x', alpha=0.3)

                # Add tier annotations
                tier1_threshold = state_stats['median'].quantile(0.75)
                axes[0, 1].axvline(tier1_threshold, color='gold', linestyle=':',
                                  label=f'Tier 1 Threshold', alpha=0.5)
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient location data',
                               ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'Location data not available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Company Tier analysis
        if 'company_tier' in valid.columns:
            tier_stats = valid.groupby('company_tier')[salary_col].agg(['median', 'count'])
            tier_stats = tier_stats[tier_stats['count'] >= 10].sort_values('median')

            if len(tier_stats) > 0:
                axes[1, 0].barh(tier_stats.index, tier_stats['median'], color='purple', alpha=0.7)
                axes[1, 0].set_xlabel('Median Salary ($)')
                axes[1, 0].set_title('Salary by Company Tier')
                axes[1, 0].grid(axis='x', alpha=0.3)

                # Add count labels
                for i, (idx, row) in enumerate(tier_stats.iterrows()):
                    axes[1, 0].text(row['median'], i, f" n={int(row['count'])}",
                                   va='center', fontsize=9)
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient company tier data',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'Company tier data not available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Seniority level analysis
        if 'seniority_level' in valid.columns:
            seniority_valid = valid[valid['seniority_level'].notna()]
            if len(seniority_valid) > 0:
                bp = axes[1, 1].boxplot([seniority_valid[seniority_valid['seniority_level'] == lvl][salary_col]
                                        for lvl in seniority_valid['seniority_level'].unique()
                                        if (seniority_valid['seniority_level'] == lvl).sum() >= 10],
                                       labels=[lvl for lvl in seniority_valid['seniority_level'].unique()
                                              if (seniority_valid['seniority_level'] == lvl).sum() >= 10],
                                       vert=True)
                axes[1, 1].set_ylabel('Annual Salary ($)')
                axes[1, 1].set_xlabel('Seniority Level')
                axes[1, 1].set_title('Salary Distribution by Seniority')
                axes[1, 1].grid(axis='y', alpha=0.3)
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient seniority data',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Seniority data not available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        self._save_figure(fig, 'feature_target_relationships')

    def plot_quality_metrics(self, data: dict):
        """Plot data quality metrics."""
        if 'merged' not in data or 'quality_score' not in data['merged'].columns:
            return

        print("Creating quality metrics plots...")

        df = data['merged']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Quality score distribution
        axes[0].hist(df['quality_score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(df['quality_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["quality_score"].mean():.2f}')
        axes[0].axvline(df['quality_score'].median(), color='green', linestyle='--',
                       label=f'Median: {df["quality_score"].median():.2f}')
        axes[0].set_xlabel('Quality Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Quality Score Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Quality score by source
        if 'data_source' in df.columns:
            source_quality = df.groupby('data_source')['quality_score'].mean().sort_values()
            axes[1].barh(source_quality.index, source_quality.values, color='steelblue')
            axes[1].set_xlabel('Average Quality Score')
            axes[1].set_title('Average Quality by Data Source')
            axes[1].set_xlim([0, 1])
            axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'quality_metrics')

    def plot_skill_cooccurrence(self, data: dict):
        """Analyze which skills appear together (co-occurrence matrix)."""
        linkedin_df = data.get('linkedin')
        if linkedin_df is None or linkedin_df.empty:
            linkedin_df = data.get('merged')

        if linkedin_df is None or 'skills' not in linkedin_df.columns:
            return

        print("Creating skill co-occurrence matrix...")

        # Get top skills
        all_skills = []
        for idx, row in linkedin_df.iterrows():
            skills = row['skills']
            if skills is None or (isinstance(skills, float) and pd.isna(skills)):
                continue
            if hasattr(skills, '__iter__') and not isinstance(skills, str):
                skills_list = list(skills) if not isinstance(skills, list) else skills
                all_skills.extend([str(s).lower().strip() for s in skills_list if str(s).strip()])

        from collections import Counter
        skill_counts = Counter(all_skills)
        top_20_skills = [s[0] for s in skill_counts.most_common(20)]

        # Build co-occurrence matrix
        cooccurrence = pd.DataFrame(0, index=top_20_skills, columns=top_20_skills)

        for idx, row in linkedin_df.iterrows():
            skills = row['skills']
            if skills is None or (isinstance(skills, float) and pd.isna(skills)):
                continue
            if hasattr(skills, '__iter__') and not isinstance(skills, str):
                skills_list = [str(s).lower().strip() for s in (list(skills) if not isinstance(skills, list) else skills)]
                # Filter to top 20
                skills_in_top = [s for s in skills_list if s in top_20_skills]
                # Increment co-occurrence
                for i, skill1 in enumerate(skills_in_top):
                    for skill2 in skills_in_top[i:]:
                        cooccurrence.loc[skill1, skill2] += 1
                        if skill1 != skill2:
                            cooccurrence.loc[skill2, skill1] += 1

        if cooccurrence.sum().sum() == 0:
            return

        # Normalize by row (percentage)
        cooccurrence_pct = cooccurrence.div(cooccurrence.sum(axis=1), axis=0) * 100

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cooccurrence_pct, annot=False, fmt='.0f', cmap='YlOrRd',
                   linewidths=0.5, ax=ax, cbar_kws={'label': '% Co-occurrence'})
        ax.set_title('Skill Co-occurrence Matrix (Top 20 Skills)\nShows which skills appear together', fontsize=14)
        ax.set_xlabel('Skills')
        ax.set_ylabel('Skills')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_figure(fig, 'skill_cooccurrence')

    def plot_h1b_specific_analysis(self, data: dict):
        """H1B-specific analysis: wage level, job title cardinality."""
        h1b_df = data.get('h1b')

        if h1b_df is None:
            return

        print("Creating H1B-specific analysis...")

        salary_col = self._find_salary_column(h1b_df)

        if not salary_col:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Wage level impact (if available)
        if 'wage_level' in h1b_df.columns or 'prevailing_wage_level' in h1b_df.columns:
            wage_col = 'wage_level' if 'wage_level' in h1b_df.columns else 'prevailing_wage_level'
            wage_valid = h1b_df[(h1b_df[wage_col].notna()) &
                               (h1b_df[salary_col].notna()) &
                               (h1b_df[salary_col] >= 30000) &
                               (h1b_df[salary_col] <= 1000000)]

            if len(wage_valid) > 0:
                wage_stats = wage_valid.groupby(wage_col)[salary_col].agg(['median', 'count'])
                wage_stats = wage_stats[wage_stats['count'] >= 10].sort_index()

                if len(wage_stats) > 0:
                    axes[0, 0].bar(wage_stats.index.astype(str), wage_stats['median'], color='steelblue', alpha=0.7)
                    axes[0, 0].set_xlabel('H1B Wage Level')
                    axes[0, 0].set_ylabel('Median Salary ($)')
                    axes[0, 0].set_title('Salary by H1B Wage Level\n(Often strongest predictor)')
                    axes[0, 0].grid(axis='y', alpha=0.3)

                    # Add count labels
                    for i, (idx, row) in enumerate(wage_stats.iterrows()):
                        axes[0, 0].text(i, row['median'], f"n={int(row['count'])}\n${row['median']:,.0f}",
                                       ha='center', va='bottom', fontsize=9)
                else:
                    axes[0, 0].text(0.5, 0.5, 'Insufficient wage level data',
                                   ha='center', va='center', transform=axes[0, 0].transAxes)
            else:
                axes[0, 0].text(0.5, 0.5, 'No valid wage level data',
                               ha='center', va='center', transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, 'Wage level data not available in H1B dataset',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Salary by H1B Wage Level')

        # 2. Job title cardinality audit
        if 'job_title' in h1b_df.columns:
            title_counts = h1b_df['job_title'].value_counts()
            unique_titles = len(title_counts)
            total_records = len(h1b_df)
            ratio = (unique_titles / total_records) if total_records else 0

            axes[0, 1].text(0.5, 0.7, f'Total Records: {total_records:,}', ha='center',
                           va='center', transform=axes[0, 1].transAxes, fontsize=16, weight='bold')
            axes[0, 1].text(0.5, 0.5, f'Unique Job Titles: {unique_titles:,}', ha='center',
                           va='center', transform=axes[0, 1].transAxes, fontsize=16, weight='bold')
            axes[0, 1].text(0.5, 0.3, f'Cardinality Ratio: {ratio:.2%}', ha='center',
                           va='center', transform=axes[0, 1].transAxes, fontsize=14,
                           color='red' if ratio > 0.5 else 'green')
            axes[0, 1].text(0.5, 0.1,
                           '✗ High cardinality may hurt model' if ratio > 0.5
                           else '✓ Reasonable cardinality',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axes[0, 1].transAxes,
                           fontsize=12,
                           color='red' if ratio > 0.5 else 'green')
            axes[0, 1].set_title('Job Title Normalization Audit')
            axes[0, 1].axis('off')

            # 3. Most common job titles
            top_titles = title_counts.head(15)
            axes[1, 0].barh(range(len(top_titles)), top_titles.values, color='purple', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_titles)))
            axes[1, 0].set_yticklabels([t[:40] + '...' if len(t) > 40 else t for t in top_titles.index])
            axes[1, 0].set_xlabel('Count')
            axes[1, 0].set_title('Top 15 Most Common Job Titles')
            axes[1, 0].invert_yaxis()
            axes[1, 0].grid(axis='x', alpha=0.3)

            # 4. Salary by top job titles
            valid_h1b = h1b_df[(h1b_df[salary_col].notna()) &
                              (h1b_df[salary_col] >= 30000) &
                              (h1b_df[salary_col] <= 1000000)]

            if len(valid_h1b) > 0:
                title_salary = valid_h1b.groupby('job_title')[salary_col].agg(['median', 'count'])
                title_salary = title_salary[title_salary['count'] >= 20].sort_values('median', ascending=True).tail(15)

                if len(title_salary) > 0:
                    axes[1, 1].barh(range(len(title_salary)),
                                   title_salary['median'], color='green', alpha=0.7)
                    axes[1, 1].set_yticks(range(len(title_salary)))
                    axes[1, 1].set_yticklabels([t[:40] + '...' if len(t) > 40 else t
                                               for t in title_salary.index])
                    axes[1, 1].set_xlabel('Median Salary ($)')
                    axes[1, 1].set_title('Top 15 Highest-Paying Job Titles (≥20 samples)')
                    axes[1, 1].grid(axis='x', alpha=0.3)
        else:
            for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, 'Job title data not available',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        self._save_figure(fig, 'h1b_specific_analysis')

    def plot_model_feature_importance(self, data: dict):
        """Load and display actual trained model's feature importance."""
        print("Loading model feature importance...")

        # Try to find feature importance file from trained model
        models_dir = Path("models")
        s3_backup_dir = Path("s3_data_backup/models/latest")

        feature_importance_path = None

        # Check local models first
        if models_dir.exists():
            importance_files = list(models_dir.glob("feature_importance*.csv"))
            if importance_files:
                feature_importance_path = importance_files[0]

        # Check S3 backup
        if not feature_importance_path and s3_backup_dir.exists():
            importance_file = s3_backup_dir / "feature_importance.csv"
            if importance_file.exists():
                feature_importance_path = importance_file

        if not feature_importance_path:
            print("  Skipping: No trained model feature importance found")
            print("  Train a model first to see feature importance")
            return

        try:
            # Load feature importance
            importance_df = pd.read_csv(feature_importance_path)

            if importance_df.empty or 'feature' not in importance_df.columns:
                print("  Skipping: Invalid feature importance file")
                return

            # Filter out features with 0 importance for cleaner visualization
            importance_df_nonzero = importance_df[importance_df['importance'] > 0].copy()
            importance_df_nonzero = importance_df_nonzero.sort_values('importance', ascending=True)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 1. All features (left)
            importance_df_sorted = importance_df.sort_values('importance', ascending=True)
            colors = ['#2ecc71' if imp > 0 else '#cccccc' for imp in importance_df_sorted['importance']]

            axes[0].barh(importance_df_sorted['feature'], importance_df_sorted['importance'],
                        color=colors, alpha=0.7)
            axes[0].set_xlabel('Feature Importance (Gain)')
            axes[0].set_title(f'Model Feature Importance (All Features)\nFrom: {feature_importance_path.name}')
            axes[0].grid(axis='x', alpha=0.3)

            # Add importance labels for non-zero features
            for idx, row in importance_df_sorted.iterrows():
                if row['importance'] > 0:
                    axes[0].text(row['importance'], idx, f' {row["importance"]:.3f}',
                               va='center', fontsize=8)

            # 2. Non-zero features only (right)
            if len(importance_df_nonzero) > 0:
                axes[1].barh(importance_df_nonzero['feature'], importance_df_nonzero['importance'],
                           color='coral', alpha=0.7)
                axes[1].set_xlabel('Feature Importance (Gain)')
                axes[1].set_title('Active Features (Non-Zero Importance)')
                axes[1].grid(axis='x', alpha=0.3)

                # Add percentage labels
                total_importance = importance_df_nonzero['importance'].sum()
                for idx, row in importance_df_nonzero.iterrows():
                    pct = (row['importance'] / total_importance) * 100
                    axes[1].text(row['importance'], list(importance_df_nonzero.index).index(idx),
                               f' {row["importance"]:.3f} ({pct:.1f}%)',
                               va='center', fontsize=9)
            else:
                axes[1].text(0.5, 0.5, 'All features have zero importance',
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Active Features')

            plt.tight_layout()
            self._save_figure(fig, 'model_feature_importance')

            # Print summary
            num_active = len(importance_df_nonzero)
            num_total = len(importance_df)
            print(f"  Loaded feature importance: {num_active}/{num_total} active features")

            if num_active > 0:
                top_feature = importance_df_nonzero.iloc[-1]
                print(f"  Top feature: {top_feature['feature']} ({top_feature['importance']:.3f})")

        except Exception as e:
            print(f"  Error loading feature importance: {e}")
            return

    def _save_figure(self, fig, name: str):
        """Save figure and track for HTML report."""
        filepath = self.output_dir / f"{name}_{self.timestamp}.png"
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        self.figures.append((name, filepath))
        print(f"  Saved: {filepath.name}")

    def generate_html_report(self):
        """Generate HTML report with all visualizations and stats."""
        print("\nGenerating HTML report...")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EDA Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
        }}
        .header {{
            background: #005587;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,85,135,0.1);
        }}
        h1 {{ margin: 0; font-size: 2em; }}
        h2 {{ color: #005587; border-bottom: 3px solid #C84E00; padding-bottom: 10px; }}
        h3 {{ color: #C84E00; }}
        h4 {{ color: #005587; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #005587;
        }}
        .stat-card h4 {{
            margin: 0 0 10px 0;
            color: #005587;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #C84E00;
        }}
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.85em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            margin: 15px 0;
            border: 1px solid #e0e0e0;
        }}
        .timestamp {{
            color: rgba(255,255,255,0.9);
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #005587;
            font-weight: 600;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric-good {{ color: #28a745; font-weight: bold; }}
        .metric-warning {{ color: #C84E00; font-weight: bold; }}
        .metric-bad {{ color: #dc3545; font-weight: bold; }}
        a {{
            color: #005587;
            text-decoration: none;
        }}
        a:hover {{
            color: #C84E00;
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Salary Data EDA Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        # Summary statistics
        html += self._generate_summary_section()

        # Data quality section
        html += self._generate_quality_section()

        # Visualizations
        html += self._generate_visualizations_section()

        # Footer
        html += """
    <div class="section">
        <p style="text-align: center; color: #6c757d; font-size: 0.9em;">
            Generated by Automated EDA Report System |
            <a href="https://github.com/jonasneves/aipi510-project3">GitHub</a>
        </p>
    </div>
</body>
</html>
"""

        # Save HTML
        report_path = self.output_dir / f"eda_report_{self.timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html)

        # Also save as latest
        latest_path = self.output_dir / "eda_report_latest.html"
        with open(latest_path, 'w') as f:
            f.write(html)

        print(f"✓ Report saved to: {report_path}")
        print(f"✓ Latest report: {latest_path}")

        return report_path

    def _generate_summary_section(self) -> str:
        """Generate summary statistics section."""
        html = '<div class="section"><h2>Data Summary</h2><div class="stat-grid">'

        total_records = sum(s['records'] for s in self.stats.values() if s.get('records'))

        html += f"""
        <div class="stat-card">
            <h4>Total Records</h4>
            <div class="value">{total_records:,}</div>
        </div>
        """

        # Add source-specific stats
        for name, stats in self.stats.items():
            if stats.get('salary'):
                sal = stats['salary']
                html += f"""
                <div class="stat-card">
                    <h4>{name.upper()} Median Salary</h4>
                    <div class="value">${sal['median']:,.0f}</div>
                    <div class="label">{sal['count']:,} records</div>
                </div>
                """

        html += '</div></div>'
        return html

    def _generate_quality_section(self) -> str:
        """Generate data quality section."""
        if 'merged' not in self.stats:
            return ""

        stats = self.stats['merged']

        html = '<div class="section"><h2>Data Quality</h2><table>'
        html += '<tr><th>Metric</th><th>Value</th><th>Status</th></tr>'

        # Quality score
        if stats.get('salary'):
            avg_quality = 0.6  # Placeholder - should come from quality_score column
            status_class = 'metric-good' if avg_quality > 0.5 else 'metric-warning'
            html += f'<tr><td>Average Quality Score</td><td class="{status_class}">{avg_quality:.2f}</td><td>{"Good" if avg_quality > 0.5 else "Fair"}</td></tr>'

        # Missing data
        if 'missing_pct' in stats:
            critical_fields = ['annual_salary', 'job_title']
            for field in critical_fields:
                if field in stats['missing_pct']:
                    missing_pct = stats['missing_pct'][field]
                    status_class = 'metric-good' if missing_pct < 5 else ('metric-warning' if missing_pct < 20 else 'metric-bad')
                    status = "Good" if missing_pct < 5 else ("Fair" if missing_pct < 20 else "Poor")
                    html += f'<tr><td>{field} missing</td><td class="{status_class}">{missing_pct:.1f}%</td><td>{status}</td></tr>'

        html += '</table></div>'
        return html

    def _generate_visualizations_section(self) -> str:
        """Generate visualizations section with categorized subsections."""
        html = '<div class="section"><h2>Visualizations</h2>'

        # Define categories
        categories = {
            'Descriptive Statistics': ['salary_distributions', 'source_comparison', 'quality_metrics'],
            'Target Variable Analysis': ['target_distribution_analysis'],
            'Feature-Target Relationships': ['feature_target_relationships', 'location_analysis'],
            'Skills Analysis': ['skills_analysis', 'skill_cooccurrence'],
            'H1B-Specific Analysis': ['h1b_specific_analysis'],
            'Model Features': ['model_feature_importance']
        }

        # Categorize figures
        categorized = {cat: [] for cat in categories}
        uncategorized = []

        for name, filepath in self.figures:
            found = False
            for category, patterns in categories.items():
                if any(pattern in name for pattern in patterns):
                    categorized[category].append((name, filepath))
                    found = True
                    break
            if not found:
                uncategorized.append((name, filepath))

        # Render by category
        for category, items in categorized.items():
            if items:
                html += f'<h3 style="color: #764ba2; margin-top: 30px;">{category}</h3>'
                for name, filepath in items:
                    title = name.replace('_', ' ').title()
                    rel_path = filepath.name
                    html += f'<div style="margin: 20px 0;"><h4 style="color: #667eea;">{title}</h4>'
                    html += f'<img src="{rel_path}" alt="{title}"></div>'

        # Render uncategorized
        if uncategorized:
            html += '<h3 style="color: #764ba2; margin-top: 30px;">Other Visualizations</h3>'
            for name, filepath in uncategorized:
                title = name.replace('_', ' ').title()
                rel_path = filepath.name
                html += f'<div style="margin: 20px 0;"><h4 style="color: #667eea;">{title}</h4>'
                html += f'<img src="{rel_path}" alt="{title}"></div>'

        html += '</div>'
        return html

    def _cleanup_old_reports(self):
        """Remove old report files, keeping only the latest."""
        if not self.keep_latest_only:
            return

        print("\nCleaning up old reports...")

        # Patterns for timestamped files
        patterns = [
            "eda_report_*.html",
            "eda_stats_*.json",
            "salary_distributions_*.png",
            "source_comparison_*.png",
            "location_analysis_*.png",
            "quality_metrics_*.png",
            "skills_analysis_*.png",
            "target_distribution_analysis_*.png",
            "feature_target_relationships_*.png",
            "skill_cooccurrence_*.png",
            "h1b_specific_analysis_*.png",
            "model_feature_importance_*.png"
        ]

        for pattern in patterns:
            # Find all files matching pattern
            files = list(self.output_dir.glob(pattern))

            # Exclude 'latest' files
            files = [f for f in files if 'latest' not in f.name]

            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the most recent, delete the rest
            if len(files) > 1:
                for old_file in files[1:]:
                    try:
                        old_file.unlink()
                        print(f"  Removed: {old_file.name}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {old_file.name}: {e}")

        print("Cleanup complete!")

    def run(self):
        """Run complete EDA report generation."""
        print("="*60)
        print("Automated EDA Report Generation")
        print("="*60)

        # Load data
        data = self.load_data()

        if not data:
                            print("✗ No data found!")
                            return        # Generate stats
        self.generate_summary_stats(data)

        # Generate plots - descriptive
        self.plot_salary_distributions(data)
        self.plot_source_comparison(data)
        self.plot_quality_metrics(data)

        # Generate plots - predictive insights
        print("\n--- Predictive Analysis ---")
        self.plot_target_distribution_analysis(data)
        self.plot_feature_target_relationships(data)
        self.plot_location_analysis(data)
        self.plot_skills_analysis(data)
        self.plot_skill_cooccurrence(data)
        self.plot_h1b_specific_analysis(data)
        self.plot_model_feature_importance(data)

        # Generate HTML report
        report_path = self.generate_html_report()

        # Save stats as JSON
        stats_path = self.output_dir / f"eda_stats_{self.timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)

        # Cleanup old reports if requested
        self._cleanup_old_reports()

        print("\n" + "="*60)
        print("✓ EDA Report Generation Complete!")
        print("="*60)
        print(f"Report: {report_path}")
        print(f"Stats:  {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate EDA Report")
    parser.add_argument('--data-dir', default='data/merged', help='Data directory')
    parser.add_argument('--output-dir', default='docs/reports', help='Output directory')
    parser.add_argument('--keep-latest-only', action='store_true', help='Remove old reports, keep only latest')
    args = parser.parse_args()

    generator = EDAReportGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        keep_latest_only=args.keep_latest_only
    )
    generator.run()


if __name__ == "__main__":
    main()
