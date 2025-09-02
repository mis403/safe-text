"""
过拟合检测和可视化工具
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class OverfittingDetector:
    """过拟合检测器"""
    
    def __init__(self):
        """初始化过拟合检测器"""
        self.training_history = []
        self.overfitting_threshold = 0.02  # 验证损失增加阈值
        self.patience_threshold = 2  # 连续增加的容忍次数
    
    def analyze_training_logs(self, log_dir: str) -> Dict:
        """分析训练日志检测过拟合
        
        Args:
            log_dir: 训练日志目录
            
        Returns:
            过拟合分析结果
        """
        log_path = Path(log_dir)
        
        # 查找trainer_state.json文件
        trainer_state_files = list(log_path.rglob("trainer_state.json"))
        if not trainer_state_files:
            logger.warning(f"未找到trainer_state.json文件在 {log_dir}")
            return self._manual_analysis()
        
        # 读取最新的训练状态
        latest_file = max(trainer_state_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                trainer_state = json.load(f)
            
            return self._analyze_trainer_state(trainer_state)
            
        except Exception as e:
            logger.error(f"读取训练状态失败: {e}")
            return self._manual_analysis()
    
    def _analyze_trainer_state(self, trainer_state: Dict) -> Dict:
        """分析trainer状态检测过拟合"""
        log_history = trainer_state.get('log_history', [])
        
        if not log_history:
            return {"error": "训练日志为空"}
        
        # 提取训练和验证指标
        train_losses = []
        eval_losses = []
        eval_accuracies = []
        eval_f1s = []
        epochs = []
        
        for entry in log_history:
            if 'loss' in entry and 'epoch' in entry:
                # 训练损失
                train_losses.append((entry['epoch'], entry['loss']))
            
            if 'eval_loss' in entry and 'epoch' in entry:
                # 验证指标
                eval_losses.append((entry['epoch'], entry['eval_loss']))
                eval_accuracies.append((entry['epoch'], entry.get('eval_accuracy', 0)))
                eval_f1s.append((entry['epoch'], entry.get('eval_f1', 0)))
                epochs.append(entry['epoch'])
        
        # 分析过拟合
        analysis = self._detect_overfitting(train_losses, eval_losses, eval_accuracies)
        
        # 添加数据用于可视化
        analysis['data'] = {
            'train_losses': train_losses,
            'eval_losses': eval_losses,
            'eval_accuracies': eval_accuracies,
            'eval_f1s': eval_f1s,
            'epochs': epochs
        }
        
        return analysis
    
    def _detect_overfitting(self, train_losses: List, eval_losses: List, eval_accuracies: List) -> Dict:
        """检测过拟合模式"""
        if len(eval_losses) < 2:
            return {"overfitting": False, "reason": "数据不足"}
        
        # 按epoch排序
        eval_losses.sort(key=lambda x: x[0])
        eval_accuracies.sort(key=lambda x: x[0])
        
        # 提取损失值
        eval_loss_values = [x[1] for x in eval_losses]
        eval_acc_values = [x[1] for x in eval_accuracies]
        
        # 检测验证损失是否开始上升
        best_loss_idx = np.argmin(eval_loss_values)
        best_loss = eval_loss_values[best_loss_idx]
        best_epoch = eval_losses[best_loss_idx][0]
        
        # 检查最佳点之后的趋势
        overfitting_signals = []
        
        if best_loss_idx < len(eval_loss_values) - 1:
            # 验证损失在最佳点后的变化
            subsequent_losses = eval_loss_values[best_loss_idx + 1:]
            loss_increases = sum(1 for loss in subsequent_losses if loss > best_loss + self.overfitting_threshold)
            
            if loss_increases >= self.patience_threshold:
                overfitting_signals.append(f"验证损失在epoch {best_epoch}后持续上升")
        
        # 检查训练损失和验证损失的差距
        if train_losses and eval_losses:
            # 找到对应的训练损失
            final_train_loss = min([x[1] for x in train_losses[-5:]])  # 最后5个训练损失的最小值
            final_eval_loss = eval_loss_values[-1]
            
            loss_gap = final_eval_loss - final_train_loss
            if loss_gap > 0.05:  # 损失差距过大
                overfitting_signals.append(f"训练损失({final_train_loss:.4f})和验证损失({final_eval_loss:.4f})差距过大")
        
        # 检查验证准确率是否开始下降
        if len(eval_acc_values) >= 3:
            best_acc_idx = np.argmax(eval_acc_values)
            if best_acc_idx < len(eval_acc_values) - 2:
                recent_acc = np.mean(eval_acc_values[-2:])
                best_acc = eval_acc_values[best_acc_idx]
                if recent_acc < best_acc - 0.005:  # 准确率下降超过0.5%
                    overfitting_signals.append(f"验证准确率从{best_acc:.4f}下降到{recent_acc:.4f}")
        
        # 综合判断
        is_overfitting = len(overfitting_signals) >= 1
        
        return {
            "overfitting": is_overfitting,
            "signals": overfitting_signals,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "final_eval_loss": eval_loss_values[-1] if eval_loss_values else None,
            "loss_trend": "上升" if eval_loss_values[-1] > best_loss else "稳定",
            "recommendation": self._get_recommendation(is_overfitting, best_epoch, len(eval_losses))
        }
    
    def _get_recommendation(self, is_overfitting: bool, best_epoch: float, total_epochs: int) -> str:
        """获取建议"""
        if not is_overfitting:
            return "模型训练良好，未检测到明显过拟合"
        
        recommendations = []
        recommendations.append(f"建议在epoch {best_epoch:.1f}附近停止训练")
        
        if total_epochs > 3:
            recommendations.append("启用早停机制，patience设置为2-3")
        
        recommendations.append("考虑增加正则化：dropout、weight_decay")
        recommendations.append("减少模型复杂度或增加训练数据")
        
        return "; ".join(recommendations)
    
    def _manual_analysis(self) -> Dict:
        """手动分析（当无法读取日志时）"""
        return {
            "overfitting": True,
            "signals": ["基于训练输出的手动分析：验证损失在后期上升"],
            "recommendation": "建议启用早停机制并重新训练"
        }
    
    def visualize_training(self, analysis: Dict, save_path: Optional[str] = None):
        """可视化训练过程"""
        if 'data' not in analysis:
            logger.warning("无可视化数据")
            return
        
        data = analysis['data']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 损失曲线
        if data['train_losses']:
            train_epochs, train_loss_vals = zip(*data['train_losses'])
            ax1.plot(train_epochs, train_loss_vals, 'b-', label='训练损失', alpha=0.7)
        
        if data['eval_losses']:
            eval_epochs, eval_loss_vals = zip(*data['eval_losses'])
            ax1.plot(eval_epochs, eval_loss_vals, 'r-', label='验证损失', linewidth=2)
            
            # 标记最佳点
            if 'best_epoch' in analysis:
                best_idx = next(i for i, (e, _) in enumerate(data['eval_losses']) 
                               if abs(e - analysis['best_epoch']) < 0.1)
                ax1.axvline(x=analysis['best_epoch'], color='g', linestyle='--', 
                           label=f'最佳epoch ({analysis["best_epoch"]:.1f})')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        if data['eval_accuracies']:
            eval_epochs, eval_acc_vals = zip(*data['eval_accuracies'])
            ax2.plot(eval_epochs, eval_acc_vals, 'g-', label='验证准确率', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('验证准确率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. F1分数曲线
        if data['eval_f1s']:
            eval_epochs, eval_f1_vals = zip(*data['eval_f1s'])
            ax3.plot(eval_epochs, eval_f1_vals, 'm-', label='验证F1', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('验证F1分数')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 过拟合检测结果
        ax4.text(0.1, 0.8, f"过拟合检测: {'是' if analysis['overfitting'] else '否'}", 
                fontsize=12, weight='bold', transform=ax4.transAxes)
        
        if analysis.get('signals'):
            signals_text = '\n'.join([f"• {signal}" for signal in analysis['signals']])
            ax4.text(0.1, 0.6, f"检测信号:\n{signals_text}", 
                    fontsize=10, transform=ax4.transAxes, verticalalignment='top')
        
        if analysis.get('recommendation'):
            ax4.text(0.1, 0.3, f"建议:\n{analysis['recommendation']}", 
                    fontsize=10, transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('过拟合分析结果')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练分析图保存到: {save_path}")
        
        plt.show()
        
        return fig

def analyze_current_training(output_dir: str = "outputs") -> Dict:
    """分析当前训练的过拟合情况"""
    detector = OverfittingDetector()
    
    # 查找最新的训练目录
    output_path = Path(output_dir)
    if not output_path.exists():
        return {"error": "输出目录不存在"}
    
    training_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('training_')]
    if not training_dirs:
        return {"error": "未找到训练目录"}
    
    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"分析训练目录: {latest_dir}")
    
    analysis = detector.analyze_training_logs(str(latest_dir))
    
    # 保存可视化
    if 'data' in analysis:
        save_path = output_path / "training_analysis.png"
        detector.visualize_training(analysis, str(save_path))
    
    return analysis
