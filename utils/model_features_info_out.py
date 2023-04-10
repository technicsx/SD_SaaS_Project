from matplotlib import pyplot
import pandas as pd


def output_overall_features_importance_diagram(importance):
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: ' + str(i) + ', Score: ' + str(v))

    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def output_top_features_diagram(dataframe, importance, is_positive, diagram_kind='barh', top_count=20,
                                target_column='is_alarm'):
    feature_importances = pd.Series(importance, index=dataframe.drop([target_column], axis=1).columns)

    feature_importances.nlargest(top_count).plot(kind=diagram_kind) \
        if is_positive else \
        feature_importances.nsmallest(top_count).plot(kind=diagram_kind)

    pyplot.title("Top " + top_count + " important features")
    pyplot.show()
