### KK ###
from import_KK import *
from data_KK import *
from preprocessing_KK import *
from description_KK import *
from evaluation_KK import *



### Date and Author: 20250611, Kyungwon Kim ###
### SHAP explanation
def explanation_SHAP_values(model, X_train, X_test, X_colname,
                            model_type='Linear'): # 'Linear', 'Tree', 'Deep'
    # Format Transformation
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values

    # SHAP model
    if model_type.lower() == 'tree':
        explainer = shap.TreeExplainer(model, data=X_train, feature_names=X_colname, approximate=True)
        shap_values_train = explainer(X_train, check_additivity=False)
        shap_values_test = explainer(X_test, check_additivity=False)
        ## Sampling for Interaction Calculation

        ## Feature Interaction Calculation
        explainer_inter = shap.TreeExplainer(model, data=None, feature_names=X_colname, approximate=False)
        shap_intervalues_train = explainer_inter.shap_interaction_values(X_train)
        shap_intervalues_test = explainer_inter.shap_interaction_values(X_test)
    else:
        explainer = shap.Explainer(model, X_train, 
                                   algorithm=model_type.lower(), feature_names=X_colname)
        shap_values_train = explainer(X_train)
        shap_values_test = explainer(X_test)
        shap_intervalues_train = None
        shap_intervalues_test = None

    return shap_values_train, shap_values_test, shap_intervalues_train, shap_intervalues_test


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP individual explanation
def explanation_SHAP_individual(shap_values, 
                                output_type='identity',    # 'identity', 'logit'
                                X_colname=None,
                                max_display=10):
    print('Individual Explanation(1 Decision Plot -> 1 Force Plot -> 1000 Force Plot)...')
    # Dimension Rearrange
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:,:,-1]
        
    # Visualization
    random_index = np.random.randint(shap_values.values.shape[0])
    shap.decision_plot(base_value=shap_values.base_values[random_index],
                       shap_values=shap_values.values[random_index],
                       features=shap_values.data[random_index],
                       feature_names=X_colname,
                       feature_display_range=slice(None, -max_display, -1),
                       link=output_type, highlight=0, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(20)          # Y축 폰트 크기
    for label in ax.get_xticklabels():
        label.set_fontsize(20)          # X축 폰트 크기
    fig.tight_layout(pad=-10) 
    plt.show()
    shap.initjs()
    display(shap.force_plot(base_value=shap_values.base_values[random_index],
                            shap_values=shap_values.values[random_index],
                            features=shap_values.data[random_index],
                            feature_names=X_colname,
                            link=output_type))


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP total explanation
def explanation_SHAP_total(shap_values, X_colname=None, max_display=10):
    print('Total Explanation(Beeswarm Plot)...')
    # Dimension Rearrange
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:,:,-1]
        
    # Beeswarm Plot
    shap.plots.beeswarm(shap_values=shap_values, max_display=max_display, show=False) # order=shap_values_train.abs.max(0)
    fig = plt.gcf()
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(20)          # Y축 폰트 크기
    for label in ax.get_xticklabels():
        label.set_fontsize(16)          # X축 폰트 크기
    fig.tight_layout(pad=-10) 
    plt.subplots_adjust(top=1.5, bottom=0.8)    # Y축 레이블 간격
    plt.show()


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP main
def explanation_SHAP(model, X_train, X_test, X_colname, 
                     model_type='Linear',    # 'Linear', 'Tree', 'Deep'
                     output_type='identity',    # 'identity', 'logit'
                     max_display=10, dependency=False):
    # Calculate SHAP
    shap_values_train, shap_values_test,\
    shap_intervalues_train, shap_intervalues_test = explanation_SHAP_values(model, X_train, X_test, 
                                                                            X_colname, model_type=model_type)
    # Individual
    print('Train Dataset:')
    explanation_SHAP_individual(shap_values_train, output_type, X_colname,
                                max_display=max_display)
    print('Test Dataset:')
    explanation_SHAP_individual(shap_values_test, output_type, X_colname,
                                max_display=max_display)
    
    # Total
    print('Train Dataset:')
    explanation_SHAP_total(shap_values_train, X_colname, max_display=max_display)
    print('Test Dataset:')
    explanation_SHAP_total(shap_values_test, X_colname, max_display=max_display)



    