#!/usr/bin/env python3
"""
SOTA Visualizer - Pipeline Edition
効率的なデータ処理とSE向けの柔軟な制御を実現

主な特徴:
- メモリ効率的なパイプライン処理
- ストレージIO最小化
- SE向けの改変しやすい設計
- マルチプロジェクト統合対応
"""

import json
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np


class SOTAVisualizer:
    """効率的なSOTA可視化パイプライン"""
    
    def __init__(self, project_root: Path, config: Optional[Dict] = None):
        """
        Args:
            project_root: プロジェクトルートパス
            config: 設定辞書（Noneの場合はデフォルト/ファイルから読み込み）
        """
        self.project_root = Path(project_root)
        self.config = config or self._load_config()
        
        # データキャッシュ（メモリ効率のため）
        self.data_cache = {}
        self.changelog_cache = {}
        
        # 出力ディレクトリ
        self.output_base = self.project_root / "User-shared/visualizations/sota"
        self.output_dirs = {
            'project': self.output_base / 'project',
            'hardware': self.output_base / 'hardware', 
            'family': self.output_base / 'family',
            'local': self.output_base / 'local'
        }
        
        # プロジェクト開始時刻
        self.project_start_time = self._get_project_start_time()
        
        # 理論性能（hardware_info.mdから読み取り）
        self.theoretical_performance = None
        
    def _load_config(self) -> Dict:
        """設定ファイル読み込み（SE制御用）"""
        config_path = self.project_root / "Agent-shared/sota_pipeline_config.json"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Config load error: {e}, using defaults")
        
        # デフォルト設定
        return {
            "pipeline": {
                "levels": ["local", "family", "hardware", "project"],  # 実行順序
                "critical_section": True,  # ロック制御
                "max_local_agents": 10,  # localの最大処理数
                "io_delay_ms": 500  # IO負荷軽減用待機時間
            },
            "dpi": {
                "local": {"linear": 60, "log": 40},
                "family": {"linear": 70, "log": 45},
                "hardware": {"linear": 80, "log": 50},
                "project": {"linear": 100, "log": 60},
                "debug": 30  # デバッグ時の統一DPI
            },
            "axes": {
                "x_options": ["time", "count", "version"],
                "y_options": ["performance", "accuracy", "efficiency"],
                "show_error_bars": True,
                "accuracy_threshold": None  # 精度フィルタ（例: 95.0）
            },
            "io_optimization": {
                "compress_level": 1,  # PNG圧縮レベル(1=最小)
                "buffer_writes": True,
                "cleanup_old_hours": 2  # 古いファイル削除
            }
        }
    
    def _get_project_start_time(self) -> datetime:
        """プロジェクト開始時刻を取得"""
        start_file = self.project_root / "Agent-shared/project_start_time.txt"
        
        if start_file.exists():
            try:
                content = start_file.read_text().strip()
                return datetime.fromisoformat(content.replace('Z', '+00:00'))
            except:
                pass
        
        # デフォルト: 現在時刻
        now = datetime.now(timezone.utc)
        start_file.parent.mkdir(parents=True, exist_ok=True)
        start_file.write_text(now.isoformat())
        return now
    
    def run(self, mode: str = 'pipeline', **params) -> bool:
        """
        メインエントリポイント
        
        Args:
            mode: 実行モード ('pipeline', 'single', 'debug', 'summary', 'export')
            **params: 追加パラメータ
        
        Returns:
            成功時True
        """
        if mode == 'summary':
            return self._run_summary_mode(**params)
        elif mode == 'export':
            return self._run_export_mode(**params)
        elif mode == 'debug':
            params['debug'] = True
            return self._run_pipeline_mode(**params)
        elif mode == 'single':
            return self._run_single_mode(**params)
        else:
            return self._run_pipeline_mode(**params)
    
    def _run_pipeline_mode(self, **params) -> bool:
        """パイプラインモード（定期実行・SE制御両対応）"""
        
        # クリティカルセクション制御
        lock_file = self.project_root / "Agent-shared/.sota_pipeline.lock"
        
        if self.config['pipeline']['critical_section'] and not params.get('force'):
            if lock_file.exists():
                age = (datetime.now() - datetime.fromtimestamp(lock_file.stat().st_mtime)).seconds
                if age < 1800:  # 30分以内なら実行中とみなす
                    print(f"Pipeline locked ({age}s ago), skipping")
                    return False
                lock_file.unlink()  # 古いロックは削除
        
        lock_file.touch()
        start_time = datetime.now()
        
        try:
            print(f"[{start_time.strftime('%H:%M:%S')}] Pipeline started")
            
            # 1. データ収集フェーズ（全ChangeLog.mdを一度だけ読み込み）
            self._collect_all_data()
            
            # 2. DPI設定
            dpi_config = self._get_dpi_config(params)
            
            # 3. 実行レベル
            levels = params.get('levels', self.config['pipeline']['levels'])
            
            generated_files = []
            
            for level in levels:
                level_start = datetime.now()
                
                if level == 'local':
                    # localは個別処理（メモリ効率）
                    files = self._process_local_level(dpi_config['local'], params)
                elif level == 'family':
                    # family（第2世代以降の融合技術）
                    files = self._process_family_level(dpi_config['family'], params)
                elif level == 'hardware':
                    # hardware（localから直接集約）
                    files = self._process_hardware_level(dpi_config['hardware'], params)
                elif level == 'project':
                    # project（全体集約）
                    files = self._process_project_level(dpi_config['project'], params)
                else:
                    continue
                
                generated_files.extend(files)
                elapsed = (datetime.now() - level_start).seconds
                print(f"  {level}: {len(files)} graphs in {elapsed}s")
                
                # IO負荷軽減
                if not params.get('no_delay'):
                    time.sleep(self.config['io_optimization'].get('io_delay_ms', 500) / 1000)
            
            total_elapsed = (datetime.now() - start_time).seconds
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {len(generated_files)} files in {total_elapsed}s")
            
            # 古いファイル削除（ストレージ管理）
            if self.config['io_optimization'].get('cleanup_old_hours'):
                self._cleanup_old_files()
            
            return True
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            lock_file.unlink(missing_ok=True)
    
    def _collect_all_data(self):
        """全ChangeLog.mdを効率的に収集"""
        self.changelog_cache = {}
        
        for changelog in self.project_root.rglob("ChangeLog.md"):
            rel_path = changelog.parent.relative_to(self.project_root)
            entries = self._parse_changelog(changelog)
            if entries:
                self.changelog_cache[str(rel_path)] = entries
        
        print(f"  Collected: {len(self.changelog_cache)} ChangeLogs, "
              f"{sum(len(e) for e in self.changelog_cache.values())} entries")
    
    def _parse_changelog(self, path: Path) -> List[Dict]:
        """ChangeLog.mdを解析（効率重視）"""
        entries = []
        
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            current_entry = {}
            
            for line in lines:
                # バージョン行
                if line.startswith('### v'):
                    if current_entry and 'performance' in current_entry:
                        entries.append(current_entry.copy())
                    current_entry = {'version': line.replace('### ', '').strip()}
                
                # 性能値抽出（複数形式対応）
                elif 'GFLOPS' in line or 'TFLOPS' in line:
                    import re
                    # "312.4 GFLOPS" や "`0.312 TFLOPS`" など
                    match = re.search(r'([\d.]+)\s*(GFLOPS|TFLOPS)', line)
                    if match:
                        value = float(match.group(1))
                        if match.group(2) == 'TFLOPS':
                            value *= 1000  # TFLOPS→GFLOPS変換
                        current_entry['performance'] = value
                
                # 生成時刻（details内、バッククォート必須）
                elif '生成時刻' in line:
                    # `2025-08-19T23:45:00Z` 形式を抽出
                    import re
                    match = re.search(r'`(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)`', line)
                    if match:
                        time_str = match.group(1)
                        try:
                            timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            current_entry['timestamp'] = timestamp
                            # 経過時間計算
                            elapsed = (timestamp - self.project_start_time).total_seconds()
                            current_entry['elapsed_seconds'] = elapsed
                        except:
                            pass
                
                # 精度情報（オプション）
                elif '精度' in line or 'accuracy' in line.lower():
                    import re
                    match = re.search(r'([\d.]+)\s*%', line)
                    if match:
                        current_entry['accuracy'] = float(match.group(1))
                
                # 誤差情報（科学記法対応）
                elif '誤差' in line or 'error' in line.lower():
                    import re
                    # "2.7e-4" や "±0.003" 形式
                    match = re.search(r'([±]?\s*[\d.]+e?[+-]?\d*)', line)
                    if match:
                        error_str = match.group(1).replace('±', '').strip()
                        try:
                            current_entry['error'] = float(error_str)
                        except:
                            pass
            
            # 最後のエントリ
            if current_entry and 'performance' in current_entry:
                entries.append(current_entry)
                
        except Exception as e:
            print(f"  Parse error {path}: {e}")
        
        return entries
    
    def _process_local_level(self, dpi_config: Dict, params: Dict) -> List[Path]:
        """localレベル処理（PGごと）"""
        generated = []
        
        # 特定エージェントDPI指定を解析
        specific_dpis = self._parse_specific_dpis(params.get('specific', ''))
        
        # ChangeLogがあるディレクトリごとに処理（localレベル = 技術ディレクトリごと）
        local_dirs = {}
        for path, entries in self.changelog_cache.items():
            if entries:
                # パスから見やすい識別子を生成（例: "intel2024/OpenMP"）
                path_parts = path.split('/')
                if len(path_parts) >= 2:
                    # 最後の2階層を使用（例: intel2024/OpenMP）
                    dir_id = '/'.join(path_parts[-2:])
                else:
                    dir_id = path_parts[-1] if path_parts else path
                
                local_dirs[dir_id] = entries
        
        # 最大処理数制限
        max_agents = params.get('max_local', self.config['pipeline']['max_local_agents'])
        
        for i, (dir_id, entries) in enumerate(list(local_dirs.items())[:max_agents]):
            # DPI決定（個別指定 or デフォルト）
            dpi = specific_dpis.get(dir_id, dpi_config['linear'])
            
            # SOTA抽出（単調増加）
            sota_entries = self._extract_sota_progression(entries)
            
            if sota_entries:
                # グラフ生成
                for x_axis in params.get('x_axes', ['time']):
                    output_path = self._generate_graph(
                        f'local/{dir_id.replace("/", "_")}',
                        sota_entries,
                        f"SOTA: {dir_id}",
                        x_axis,
                        dpi,
                        params
                    )
                    if output_path:
                        generated.append(output_path)
        
        return generated
    
    def _process_hardware_level(self, dpi_config: Dict, params: Dict) -> List[Path]:
        """hardwareレベル処理（localから集約）"""
        generated = []
        
        # hardware階層を識別
        hardware_groups = {}  # コンパイラごと（single-node/gcc11.3.0など）
        hardware_merged = {}  # ハードウェア全体（single-nodeなど）
        
        for path, entries in self.changelog_cache.items():
            # hardware階層を判定
            hw_key = self._extract_hardware_key(path)
            if hw_key:
                # コンパイラごとのグループ
                if hw_key not in hardware_groups:
                    hardware_groups[hw_key] = []
                hardware_groups[hw_key].extend(entries)
                
                # ハードウェア全体のグループ（コンパイラ統合）
                hw_base = hw_key.split('/')[0]  # single-node部分のみ
                if hw_base not in hardware_merged:
                    hardware_merged[hw_base] = []
                hardware_merged[hw_base].extend(entries)
        
        # コンパイラごとのグラフ生成
        for hw_key, all_entries in hardware_groups.items():
            # 時系列でSOTA更新
            sota_entries = self._aggregate_sota_by_time(all_entries)
            
            if sota_entries:
                for x_axis in params.get('x_axes', ['time']):
                    output_path = self._generate_graph(
                        f'hardware/{hw_key.replace("/", "_")}',
                        sota_entries,
                        f"SOTA: {hw_key}",
                        x_axis,
                        dpi_config['linear'],
                        params
                    )
                    if output_path:
                        generated.append(output_path)
        
        # ハードウェア全体（コンパイラ統合）のグラフ生成
        for hw_base, all_entries in hardware_merged.items():
            # 時系列でSOTA更新
            sota_entries = self._aggregate_sota_by_time(all_entries)
            
            if sota_entries:
                for x_axis in params.get('x_axes', ['time']):
                    output_path = self._generate_graph(
                        f'hardware/{hw_base}_all',
                        sota_entries,
                        f"SOTA: {hw_base} (All Compilers)",
                        x_axis,
                        dpi_config['linear'],
                        params
                    )
                    if output_path:
                        generated.append(output_path)
        
        return generated
    
    def _process_project_level(self, dpi_config: Dict, params: Dict) -> List[Path]:
        """projectレベル処理（全体集約）"""
        generated = []
        
        # 全エントリを時系列で集約
        all_entries = []
        for entries in self.changelog_cache.values():
            all_entries.extend(entries)
        
        # SOTA更新履歴
        sota_entries = self._aggregate_sota_by_time(all_entries)
        
        if sota_entries:
            for x_axis in params.get('x_axes', ['time', 'count']):
                for log_scale in [False, True]:
                    dpi = dpi_config['log' if log_scale else 'linear']
                    
                    output_path = self._generate_graph(
                        f'project/sota_project_{x_axis}{"_log" if log_scale else ""}',
                        sota_entries,
                        "SOTA: Project Overall",
                        x_axis,
                        dpi,
                        params,
                        log_scale=log_scale
                    )
                    if output_path:
                        generated.append(output_path)
        
        return generated
    
    def _process_family_level(self, dpi_config: Dict, params: Dict) -> List[Path]:
        """familyレベル処理（第2世代以降の融合技術とその親技術）"""
        generated = []
        
        # family判定（OpenMP_MPI, OpenMP_AVX2など）
        family_found = set()
        
        for path in self.changelog_cache.keys():
            # アンダースコアを含む技術名を検出
            if '_' in path:
                parts = path.split('/')
                for part in parts:
                    if '_' in part and any(tech in part for tech in ['OpenMP', 'MPI', 'CUDA', 'AVX']):
                        family_found.add(part)
                        break
        
        # 各familyで処理（親技術も含めて）
        for family_key in family_found:
            # 親技術を特定（例：OpenMP_MPI → ['OpenMP', 'MPI']）
            parent_techs = family_key.split('_')
            
            # 関連する全データを収集
            multi_series_data = {}
            
            # 1. family自体のデータ
            for path, entries in self.changelog_cache.items():
                if family_key in path:
                    # パスから識別名を生成（例：intel2024/OpenMP_MPI）
                    path_parts = path.split('/')
                    if len(path_parts) >= 2:
                        series_key = '/'.join(path_parts[-2:])
                    else:
                        series_key = path_parts[-1] if path_parts else path
                    
                    multi_series_data[series_key] = entries
            
            # 2. 親技術のデータも収集
            for parent_tech in parent_techs:
                for path, entries in self.changelog_cache.items():
                    # 親技術の単独ディレクトリを探す（_を含まない）
                    path_parts = path.split('/')
                    for part in path_parts:
                        if part == parent_tech:  # 完全一致で親技術
                            if len(path_parts) >= 2:
                                series_key = '/'.join(path_parts[-2:])
                            else:
                                series_key = path_parts[-1] if path_parts else path
                            
                            if series_key not in multi_series_data:
                                multi_series_data[series_key] = entries
                            break
            
            # 複数系列のグラフを生成
            if multi_series_data:
                output_path = self._generate_multi_series_graph(
                    f'family/{family_key}',
                    multi_series_data,
                    f"Family: {family_key}",
                    'time',
                    dpi_config['linear'],
                    params
                )
                if output_path:
                    generated.append(output_path)
        
        return generated
    
    def _generate_graph(self, name: str, entries: List[Dict], title: str, 
                       x_axis: str, dpi: int, params: Dict, log_scale: bool = False) -> Optional[Path]:
        """グラフ生成（IO最適化版）"""
        if not entries:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # データ準備
            if x_axis == 'time':
                # elapsed_secondsがないエントリを検出
                missing_time = [e.get('version', f'unknown_{i}') for i, e in enumerate(entries) if 'elapsed_seconds' not in e]
                if missing_time:
                    print(f"  ⚠️ Warning: ChangeLogに生成時刻が不足: {', '.join(missing_time)}")
                    print(f"     {len(missing_time)}個のエントリを除外してグラフ生成")
                    # 有効なエントリのみ使用
                    entries = [e for e in entries if 'elapsed_seconds' in e]
                
                if not entries:
                    print(f"  ❌ Error: 時間情報が1つもありません。このグラフをスキップ")
                    return None
                
                # 時系列順にソート（重要！）
                entries = sorted(entries, key=lambda e: e['elapsed_seconds'])
                
                x_data = [e['elapsed_seconds'] / 60 for e in entries]  # 分単位
                x_label = 'Time (minutes from start)'
                
                # 時間スケール調整（tick数問題対策）
                max_time = max(e['elapsed_seconds'] for e in entries)
                if max_time < 7200:  # 2時間未満
                    x_label = 'Time (minutes from start)'
                    x_formatter = lambda x, pos: f'{x:.0f}m'
                elif max_time < 86400:  # 24時間未満
                    x_data = [e['elapsed_seconds'] / 3600 for e in entries]
                    x_label = 'Time (hours from start)'
                    x_formatter = lambda x, pos: f'{x:.1f}h'
                else:  # 1日以上
                    x_data = [e['elapsed_seconds'] / 86400 for e in entries]
                    x_label = 'Time (days from start)'
                    x_formatter = lambda x, pos: f'{x:.1f}d'
                
                ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
                
            elif x_axis == 'count':
                x_data = list(range(1, len(entries) + 1))
                x_label = 'Generation Count'
                
            elif x_axis == 'version':
                x_data = list(range(len(entries)))
                x_label = 'Version'
                # バージョンラベル設定
                ax.set_xticks(x_data)
                ax.set_xticklabels([e.get('version', f'v{i}') for i, e in enumerate(entries)], 
                                   rotation=45)
            else:
                x_data = list(range(len(entries)))
                x_label = x_axis.capitalize()
            
            y_data = [e['performance'] for e in entries]
            
            # 精度フィルタリング
            if params.get('accuracy_threshold'):
                filtered = [(x, y, e) for x, y, e in zip(x_data, y_data, entries)
                           if e.get('accuracy', 100) >= params['accuracy_threshold']]
                if filtered:
                    x_data, y_data, entries = zip(*filtered)
            
            # プロット（階段状、青系の色）
            ax.step(x_data, y_data, 'b-', where='post', linewidth=2, label='SOTA', alpha=0.8)
            ax.plot(x_data, y_data, 'bo', markersize=6, alpha=0.8)
            
            # 誤差バー（あれば）
            if self.config['axes']['show_error_bars'] and any('error' in e for e in entries):
                yerr = [e.get('error', 0) for e in entries]
                ax.errorbar(x_data, y_data, yerr=yerr, fmt='none', ecolor='gray', alpha=0.5)
            
            # 理論性能線（あれば）
            if self.theoretical_performance and not params.get('no_theoretical'):
                ax.axhline(y=self.theoretical_performance, color='gray', 
                          linestyle='--', alpha=0.5, label='Theoretical')
            
            # tick数制限（MAXTICKS対策）
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
            
            # スケール設定
            if log_scale:
                ax.set_yscale('log')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('Performance (GFLOPS)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 出力パス
            output_dir = self.output_base / name.rsplit('/', 1)[0]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{name.rsplit('/', 1)[-1]}.png"
            
            # 保存（圧縮最小化）
            # compress_levelはmatplotlib 3.8+のみ対応
            try:
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                           compress_level=self.config['io_optimization']['compress_level'])
            except TypeError:
                # 古いmatplotlibではcompress_level未対応
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"  Graph error {name}: {e}")
            plt.close()
            return None
    
    def _generate_multi_series_graph(self, name: str, multi_series_data: Dict[str, List[Dict]], 
                                    title: str, x_axis: str, dpi: int, params: Dict) -> Optional[Path]:
        """複数系列のグラフ生成（family用）"""
        if not multi_series_data:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # matplotlibのデフォルトカラーサイクルを使用
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            # 各系列をプロット
            for idx, (series_key, entries) in enumerate(multi_series_data.items()):
                # SOTA進行を抽出
                sota_entries = self._extract_sota_progression(entries)
                
                if not sota_entries:
                    continue
                
                # elapsed_secondsがあるエントリのみ
                valid_entries = [e for e in sota_entries if 'elapsed_seconds' in e]
                if not valid_entries:
                    continue
                
                # 時系列順にソート
                valid_entries = sorted(valid_entries, key=lambda e: e['elapsed_seconds'])
                
                # データ準備
                x_data = [e['elapsed_seconds'] / 60 for e in valid_entries]  # 分単位
                y_data = [e['performance'] for e in valid_entries]
                
                # プロット（階段状、色は自動割り当て）
                color = colors[idx % len(colors)]
                ax.step(x_data, y_data, where='post', linewidth=2, 
                       label=series_key, color=color, alpha=0.8)
                ax.plot(x_data, y_data, 'o', markersize=4, color=color, alpha=0.8)
            
            # 軸設定
            ax.set_xlabel('Time (minutes from start)')
            ax.set_ylabel('Performance (GFLOPS)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # tick数制限
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
            
            # 出力パス
            output_dir = self.output_base / name.rsplit('/', 1)[0]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{name.rsplit('/', 1)[-1]}.png"
            
            # 保存
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"  Multi-series graph error {name}: {e}")
            plt.close()
            return None
    
    def _extract_sota_progression(self, entries: List[Dict]) -> List[Dict]:
        """SOTA更新のみ抽出（単調増加）"""
        if not entries:
            return []
        
        # elapsed_secondsがあるエントリのみでソート（重要！）
        valid_entries = [e for e in entries if 'elapsed_seconds' in e]
        if not valid_entries:
            # タイムスタンプがない場合は元の順序を保持
            valid_entries = entries
        
        # タイムスタンプでソート
        sorted_entries = sorted(valid_entries, key=lambda e: e.get('elapsed_seconds', float('inf')))
        
        sota = []
        max_perf = 0
        
        for entry in sorted_entries:
            perf = entry.get('performance', 0)
            if perf > max_perf:
                max_perf = perf
                sota.append(entry)
        
        return sota
    
    def _aggregate_sota_by_time(self, entries: List[Dict]) -> List[Dict]:
        """時系列でSOTA集約"""
        if not entries:
            return []
        
        # elapsed_secondsがあるエントリのみでソート
        valid_entries = [e for e in entries if 'elapsed_seconds' in e]
        if not valid_entries:
            # タイムスタンプがない場合は元の順序を保持
            valid_entries = entries
        
        # タイムスタンプでソート
        sorted_entries = sorted(valid_entries, key=lambda e: e.get('elapsed_seconds', float('inf')))
        
        sota = []
        max_perf = 0
        
        for entry in sorted_entries:
            perf = entry.get('performance', 0)
            if perf > max_perf:
                max_perf = perf
                # 集約エントリ作成
                sota_entry = entry.copy()
                sota_entry['generation_count'] = len(sota) + 1
                sota.append(sota_entry)
        
        return sota
    
    def _extract_agent_id(self, path: str) -> Optional[str]:
        """パスからエージェントID抽出（PG1.2形式対応）"""
        import re
        # PG1, PG1.2, PG10.3などに対応
        match = re.search(r'PG\d+(?:\.\d+)?', path)
        return match.group() if match else None
    
    def _extract_hardware_key(self, path: str) -> Optional[str]:
        """パスからhardwareキー抽出"""
        # single-node/gcc11.3.0 形式を検出
        parts = path.split('/')
        
        # hardware階層のパターン
        hw_patterns = ['single-node', 'multi-node', 'gpu-cluster']
        
        for i, part in enumerate(parts):
            if part in hw_patterns and i + 1 < len(parts):
                # 次の要素がコンパイラ/モジュール
                return f"{part}/{parts[i+1]}"
        
        return None
    
    def _parse_specific_dpis(self, specific_str: str) -> Dict[str, int]:
        """特定エージェントDPI指定を解析
        
        形式: "PG1.2:120,PG2:80,SE1:100"
        """
        result = {}
        
        if not specific_str:
            return result
        
        for item in specific_str.split(','):
            if ':' in item:
                agent, dpi = item.split(':', 1)
                agent = agent.strip()
                try:
                    result[agent] = int(dpi)
                except:
                    pass
        
        return result
    
    def _get_dpi_config(self, params: Dict) -> Dict:
        """DPI設定取得"""
        if params.get('debug'):
            # デバッグモード
            debug_dpi = self.config['dpi'].get('debug', 30)
            return {
                'local': {'linear': debug_dpi, 'log': debug_dpi - 5},
                'family': {'linear': debug_dpi, 'log': debug_dpi - 5},
                'hardware': {'linear': debug_dpi + 10, 'log': debug_dpi},
                'project': {'linear': debug_dpi + 20, 'log': debug_dpi + 10}
            }
        
        return self.config['dpi']
    
    def _cleanup_old_files(self):
        """古いグラフファイル削除（ストレージ管理）"""
        max_age_hours = self.config['io_optimization'].get('cleanup_old_hours', 2)
        
        if max_age_hours <= 0:
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed = 0
        
        for png in self.output_base.rglob("*.png"):
            # milestoneは削除しない
            if 'milestone' in png.name:
                continue
            
            if png.stat().st_mtime < cutoff_time:
                png.unlink()
                removed += 1
        
        if removed > 0:
            print(f"  Cleaned up {removed} old files")
    
    def _run_summary_mode(self, **params) -> bool:
        """サマリーモード（グラフ生成なし、データ確認のみ）"""
        print("=" * 60)
        print("SOTA Data Summary")
        print("=" * 60)
        
        # データ収集
        self._collect_all_data()
        
        # 統計表示
        total_entries = sum(len(e) for e in self.changelog_cache.values())
        print(f"\nTotal: {len(self.changelog_cache)} ChangeLogs, {total_entries} entries")
        
        # レベル別サマリー
        print("\n[LOCAL]")
        pg_count = sum(1 for p in self.changelog_cache.keys() if 'PG' in p)
        print(f"  PG agents: {pg_count}")
        
        # 最新性能TOP5
        print("\n[TOP PERFORMANCE]")
        all_perfs = []
        for path, entries in self.changelog_cache.items():
            if entries:
                latest = max(entries, key=lambda e: e.get('performance', 0))
                all_perfs.append((path, latest.get('performance', 0)))
        
        for path, perf in sorted(all_perfs, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {path}: {perf:.1f} GFLOPS")
        
        # tick数チェック
        print("\n[TICK CHECK]")
        max_time = max((e.get('elapsed_seconds', 0) for entries in self.changelog_cache.values() 
                       for e in entries), default=0)
        
        if max_time > 0:
            print(f"  Max elapsed: {max_time/3600:.1f} hours")
            estimated_ticks = int(max_time / 60)
            print(f"  Estimated ticks (minutes): {estimated_ticks}")
            
            if estimated_ticks > 1000:
                print(f"  ⚠️ WARNING: Would exceed MAXTICKS!")
                print(f"  ✅ Fix: Dynamic time units + MaxNLocator(nbins=15)")
        
        return True
    
    def _run_export_mode(self, **params) -> bool:
        """エクスポートモード（マルチプロジェクト統合用）"""
        
        # データ収集
        self._collect_all_data()
        
        # エクスポートディレクトリ
        export_dir = self.project_root / "Agent-shared/exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"sota_export_{timestamp}.json"
        
        # エクスポートデータ構築
        export_data = {
            'project': str(self.project_root.name),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'start_time': self.project_start_time.isoformat(),
            'data': {},
            'metadata': {
                'total_changelogs': len(self.changelog_cache),
                'total_entries': sum(len(e) for e in self.changelog_cache.values()),
                'config': self.config
            }
        }
        
        # データ変換（datetime対応）
        for path, entries in self.changelog_cache.items():
            export_data['data'][path] = [
                {k: (v.isoformat() if isinstance(v, datetime) else v)
                 for k, v in entry.items()}
                for entry in entries
            ]
        
        # JSON保存
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported to: {export_path}")
        print(f"   Size: {export_path.stat().st_size / 1024:.1f} KB")
        
        return True
    
    def _run_single_mode(self, **params) -> bool:
        """単一グラフ生成モード（デバッグ・個別確認用）"""
        
        level = params.get('level', 'project')
        specific = params.get('specific')
        
        # データ収集
        self._collect_all_data()
        
        # DPI設定
        dpi = params.get('dpi', 100)
        
        if level == 'local' and specific:
            # 特定PGのみ
            for path, entries in self.changelog_cache.items():
                if specific in path:
                    sota = self._extract_sota_progression(entries)
                    if sota:
                        output = self._generate_graph(
                            f'local/{specific}',
                            sota,
                            f"SOTA: {specific}",
                            params.get('x_axis', 'time'),
                            dpi,
                            params
                        )
                        if output:
                            print(f"✅ Generated: {output}")
                            return True
        
        print("No data found for specified criteria")
        return False


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        description='SOTA Visualizer - Efficient Pipeline Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 通常のパイプライン実行（定期実行用）
  python sota_visualizer.py
  
  # デバッグモード（低解像度）
  python sota_visualizer.py --debug
  
  # サマリー表示（グラフ生成なし）
  python sota_visualizer.py --summary
  
  # 特定PGのみ高解像度
  python sota_visualizer.py --specific PG1.2:150
  
  # データエクスポート
  python sota_visualizer.py --export
  
  # SEカスタム実行
  python sota_visualizer.py --levels local,project --dpi 80
        """
    )
    
    # モード選択
    parser.add_argument('--pipeline', action='store_true', default=True,
                       help='Pipeline mode (default)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with low DPI')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary without generating graphs')
    parser.add_argument('--export', action='store_true',
                       help='Export data for multi-project analysis')
    parser.add_argument('--single', action='store_true',
                       help='Single graph generation mode')
    
    # パイプライン制御
    parser.add_argument('--levels', type=str,
                       help='Comma-separated levels (e.g., local,hardware,project)')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even if locked')
    parser.add_argument('--no-delay', action='store_true',
                       help='No IO delay between levels')
    
    # グラフ制御
    parser.add_argument('--specific', type=str,
                       help='Specific agents with DPI (e.g., PG1.2:120,PG2:80)')
    parser.add_argument('--x-axis', type=str, default='time',
                       choices=['time', 'count', 'version'],
                       help='X-axis type')
    parser.add_argument('--dpi', type=int,
                       help='Override default DPI')
    parser.add_argument('--accuracy-threshold', type=float,
                       help='Filter by accuracy (e.g., 95.0)')
    parser.add_argument('--no-theoretical', action='store_true',
                       help='Hide theoretical performance line')
    
    # レベル指定（単一モード用）
    parser.add_argument('--level', type=str, default='project',
                       choices=['local', 'family', 'hardware', 'project'],
                       help='Level for single mode')
    
    args = parser.parse_args()
    
    # プロジェクトルート検索
    current = Path.cwd()
    project_root = None
    
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            project_root = current
            break
        current = current.parent
    
    if not project_root:
        print("Error: Could not find project root (CLAUDE.md)")
        sys.exit(1)
    
    # Visualizer作成
    visualizer = SOTAVisualizer(project_root)
    
    # パラメータ構築
    params = {
        'force': args.force,
        'no_delay': args.no_delay,
        'specific': args.specific,
        'x_axis': args.x_axis,
        'accuracy_threshold': args.accuracy_threshold,
        'no_theoretical': args.no_theoretical
    }
    
    if args.levels:
        params['levels'] = args.levels.split(',')
    
    if args.dpi:
        params['dpi'] = args.dpi
    
    # モード判定と実行
    if args.summary:
        success = visualizer.run('summary', **params)
    elif args.export:
        success = visualizer.run('export', **params)
    elif args.debug:
        success = visualizer.run('debug', **params)
    elif args.single:
        params['level'] = args.level
        success = visualizer.run('single', **params)
    else:
        success = visualizer.run('pipeline', **params)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()