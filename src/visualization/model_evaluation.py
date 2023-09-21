from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay, mean_absolute_error, mean_squared_error, r2_score, log_loss, PrecisionRecallDisplay, accuracy_score, brier_score_loss, recall_score, precision_score, f1_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
from pandas.api.types import is_numeric_dtype

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import shap

class ModelEvaluator():
    """ Model agnostic model evaluation class.
    """
    def __init__(self, model, data, actual_name, expected_name, compare_name = None):
        """ Given a model, data, actual, expected and optionally comparison data.
            Sets up easy evaluation methods.

        Args:
            model: Can be any model class.
            data (Dataframe): Dataframe to evaluate model on.
            actual_name (Str): Column name with actual values.
            expected_name (Str): Column name with expected values.
            compare_name (Str, optional): Column name with comparison values. Defaults to None.
        """
        self.model = model
        self.data: pd.DataFrame = data
        self.actual_name: str = actual_name
        self.expected_name: str = expected_name
        self.compare_name: str = compare_name
        
        self.actual = self.data[self.actual_name]
        self.expected = self.data[self.expected_name]
        if self.compare_name is not None:
            self.compare = self.data[self.compare_name]
            
    def plot_ave(self):
        """Plot actual vs. predicted values"""
                
        # Plot actual vs. predicted values
        plt.scatter(self.actual, self.expected)
        plt.plot([0, max(self.actual)], [0, max(self.actual)], 'r--')
        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")
        plt.show()
        
    def plot_distribution(self, compare=False):
        """Plot actual vs. predicted values"""
                
        # Plot actual vs. predicted values
        fig = sns.kdeplot(self.actual, shade=True, color="r")
        fig = sns.kdeplot(self.expected, shade=True, color="b")
        plt.legend(labels = ["Actual", "Expected"])

        if compare:
            fig = sns.kdeplot(self.compare, shade=True, color="g")
            plt.legend(labels = ["Actual", "Expected", "Comparison"])
            
        plt.xlabel(self.actual_name)
        plt.show()
        
    def _get_feature_plot_data(self, feature):
        """ Aggregates actual, expected and comparison columns by specified feature.
            For numeric continuous features, creates bins.

        Args:
            feature (Str): Feature to plot.

        Returns:
            Dataframe: Aggregated data by feature.
        """
        if self.compare_name is not None:
            plot_dict = {
                'actual':self.actual,
                'expected':self.expected,
                'compare':self.compare,
                'feature':self.data[feature]
                }
        else:
            plot_dict = {
                'actual':self.actual,
                'expected':self.expected,
                'feature':self.data[feature]
                }
        plot_data = pd.DataFrame(plot_dict)

        if is_numeric_dtype(plot_data['feature']) & (len(np.unique(plot_data['feature'])) > 50):
            bins = 10
            edges = np.linspace(plot_data['feature'].min(), plot_data['feature'].max(), bins+1).astype(float).round(5)
            labels = [f'({edges[i]}, {edges[i+1]}]' for i in range(bins)]
            plot_data['feature'] = pd.cut(plot_data['feature'], bins = bins, labels = labels)
            
        if self.compare_name is not None:
            feature_plot_data = plot_data.groupby('feature').agg(
                actual = ('actual', 'mean'),
                expected = ('expected', 'mean'),
                compare = ('compare', 'mean'),
                exposure = ('actual', 'size'),
                ).reset_index()
        else:
            feature_plot_data = plot_data.groupby('feature').agg(
                actual = ('actual', 'mean'),
                expected = ('expected', 'mean'),
                exposure = ('actual', 'size'),
                ).reset_index()
        
        return feature_plot_data
    
    def plot_feature_ave(self, feature):
        """ Plots Actual v Expected (v Comparison) for feature.

        Args:
            feature (Str): Feature to plot.
        """
        
        feature_plot_data = self._get_feature_plot_data(feature)
    
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.bar(feature_plot_data['feature'],feature_plot_data['exposure'], alpha = 0.5)
        ax2.plot(feature_plot_data['feature'], feature_plot_data['actual'], label = "Actual", color = "r")
        ax2.plot(feature_plot_data['feature'], feature_plot_data['expected'], label = "Expected", color = "green")
        if self.compare_name is not None:
            ax2.plot(feature_plot_data['feature'], feature_plot_data['compare'], label = "Compare", color = "blue")

        ax1.set_xlabel(feature)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            
        ax1.set_ylabel("Exposure", fontsize=14)
        ax2.set_ylabel(self.actual_name, fontsize=14)

        ax2.legend()

        fig.suptitle("Actual v Expected: " + feature, fontsize=20)
        fig.show()
        
    def _get_double_lift_chart_data(self):
        """ Aggregates actual, expected and comparison columns into Ventiles.
            Sorted from low to high by expected / comparison ratio.

        Returns:
            Dataframe: Aggregated data by expected to comparison ratio.
        """
                
        plot_dict = {
            'actual':self.actual,
            'expected':self.expected,
            'compare':self.compare
            }
        plot_data = pd.DataFrame(plot_dict)

        plot_data['pred_ratio'] = plot_data['expected'] / plot_data['compare']
        plot_data = plot_data.sort_values(by = 'pred_ratio')

        plot_data['ventiles'] = pd.cut(plot_data['pred_ratio'], bins = 20, labels = list(range(1,21)))

        double_lift_data = plot_data.groupby('ventiles').agg(
            actual = ('actual', 'mean'),
            expected = ('expected', 'mean'),
            compare = ('compare', 'mean'),
            exposure = ('actual', 'size')
            ).reset_index()

        double_lift_data['expected_rescale'] = double_lift_data['expected'] / double_lift_data['actual']
        double_lift_data['compare_rescale'] = double_lift_data['compare'] / double_lift_data['actual']
        double_lift_data['actual_rescale'] = 1
        
        return double_lift_data
    
    def plot_double_lift_chart(self):
        """ Plots double lift chart.
        """
    
        double_lift_data = self._get_double_lift_chart_data()

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.bar(double_lift_data['ventiles'],double_lift_data['exposure'], alpha = 0.5)
        ax2.plot(double_lift_data['ventiles'], double_lift_data['actual_rescale'], label = "Actual", color = "r")
        ax2.plot(double_lift_data['ventiles'], double_lift_data['expected_rescale'], label = "Expected", color = "green")
        ax2.plot(double_lift_data['ventiles'], double_lift_data['compare_rescale'], label = "Compare", color = "blue")

        ax1.set_xlabel("Ventiled - Exposure")
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            
        ax1.set_ylabel("Number of Games", fontsize=14)
        ax2.set_ylabel("Actual to Expected Ratio", fontsize=14)

        ax2.legend()

        fig.suptitle("Double Lift Chart", fontsize=20)
        fig.show()

    
class XGBModelEvaluator(ModelEvaluator):
    """XGBoost specific model evaluation class.
    """
    def __init__(self, model, data, actual_name, expected_name, compare_name = None):
        super().__init__(model, data, actual_name, expected_name, compare_name)
    
        self.feature_names = list(self.model.feature_names_in_)
        self.shap_values = None
    
    def plot_feature_importance(self, max_num_features = 20, importance_type = "total_gain"):
        """Plot feature importance for the model"""
        xgb.plot_importance(self.model, max_num_features = max_num_features, importance_type = importance_type)
         
    def _get_shap_values(self, sample = 10000):
        """Gets SHAP values for XGBoost model.
        """
        self.sample_data = self.data[self.feature_names].sample(sample)
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(self.sample_data)

    def plot_shap_summary_plot(self, max_display=10, sample = 10000):
        """Plot SHAP values for tree-based and other models"""
        if not(self.shap_values):
            self._get_shap_values(sample=sample)
        shap.summary_plot(self.shap_values, self.sample_data, max_display = max_display)
        
    def get_ranked_feature_importance(self):
        """ For XGBoost model, ranks features by average SHAP value.

        Returns:
            List: Ranked list of importance features.
        """
        
        if not(self.shap_values):
            self._get_shap_values()    
        
        vals= np.abs(self.shap_values.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.feature_names, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)

        return list(feature_importance['col_name'])
        
    def plot_pdp(self, feature_list, target=None):
        """Plot partial dependence plot for a given feature"""
        PartialDependenceDisplay.from_estimator(self.model, self.data[self.feature_names], feature_list, target=target)
        
    def plot_ice(self, feature_list):
        """Plot individual conditional expectation (ICE) plot for a given feature"""
        PartialDependenceDisplay.from_estimator(self.model, self.data[self.feature_names], feature_list, kind="both")

    
class XGBClassifierEvaluator(XGBModelEvaluator):
    
    def __init__(self, model, data, actual_name, expected_name, compare_name=None, expected_label_name=None, compare_label_name=None):
        super().__init__(model, data, actual_name, expected_name, compare_name)
        
        self.expected_label_name: str = expected_label_name
        self.compare_label_name: str = compare_label_name
        if expected_label_name is not None:
            self.expected_label = self.data[expected_label_name]
        if compare_label_name is not None:
            self.compare_label = self.data[compare_label_name]       
    
    def get_log_loss(self):
        """Return the logloss for binary classification"""
        print("Expected Log-Loss: \t{:.4f}".format(log_loss(self.actual, self.expected)))
        if self.compare_name is not None:
            print("Compare Log-Loss: \t{:.4f}".format(log_loss(self.actual, self.compare)))
            return log_loss(self.actual, self.expected), log_loss(self.actual, self.expected)
        return log_loss(self.actual, self.expected)

    def get_confusion_matrix(self):
        """Return the confusion matrix for binary classification"""
        return confusion_matrix(self.actual, self.expected_label)

    def get_roc_curve(self):
        """Return the ROC curve for binary classification"""
        fpr, tpr, thresholds = roc_curve(self.actual, self.expected)
        return fpr, tpr, thresholds
    
    def get_auc_score(self):
        """Return the AUC score for binary classification"""
        if self.compare_name is not None:
            return roc_auc_score(self.actual, self.expected_label), roc_auc_score(self.actual, self.compare_label)
        else:
            return roc_auc_score(self.actual, self.expected_label)
        
    
    def display_confusion_matrix(self):
        """ Display confusion matrix"""
        return ConfusionMatrixDisplay.from_predictions(self.actual, self.expected_label, cmap="Blues", normalize="all")
    
    def plot_roc_curve(self):
        """ Plot the ROC curve for binary classification"""
        return RocCurveDisplay.from_predictions(self.actual, self.expected)
     
    def plot_prauc_curve(self):
        """Plot the PR-AUC curve"""
        return PrecisionRecallDisplay.from_predictions(self.actual, self.expected) 
           
    def get_brier_score_loss(self):
        """Return the brier loss score"""
        print("Expected Brier Score: \t{:.4f}".format(brier_score_loss(self.actual, self.expected)))
        if self.compare_name is not None:
            print("Compare Brier Score: \t{:.4f}".format(brier_score_loss(self.actual, self.compare)))
            return brier_score_loss(self.actual, self.expected), brier_score_loss(self.actual, self.compare)
        return brier_score_loss(self.actual, self.expected)
    
    def get_accuracy(self):
        """Return the accuracy."""
        return accuracy_score(self.actual, self.expected_label)
    
    def get_recall(self):
        """Return the recall."""
        return recall_score(self.actual, self.expected_label)
    
    def get_precision(self):
        """Return the precision."""
        return precision_score(self.actual, self.expected_label)
        
    def get_f1_score(self):
        """Return the f1 score."""
        return f1_score(self.actual, self.expected_label, average="binary")
    
    def display_calibration_curve(self, nbins=10):
        """ Plot calibration curve for binary classifier """
        return CalibrationDisplay.from_predictions(self.actual, self.expected, n_bins=nbins)
        
    
    
class XGBRegressorEvaluator(XGBModelEvaluator):

    def get_mae(self, compare=False):
        """Return the mean absolute error for regression"""
        if compare:
            return mean_absolute_error(self.actual, self.compare)
        else:
            return mean_absolute_error(self.actual, self.expected)
    
    def get_mse(self, compare=False):
        """Return the mean squared error for regression"""
        if compare:
            return mean_squared_error(self.actual, self.compare)
        else:
            return mean_squared_error(self.actual, self.expected)
    
    def get_r2_score(self, compare=False):
        """Return the R-squared score for regression"""
        if compare:
            return r2_score(self.actual, self.compare)
        else:
            return r2_score(self.actual, self.expected)
    
