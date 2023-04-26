from analyzer import Analyzer

if __name__ == '__main__':

    analyzer = Analyzer(
        mode='opt', results_path='output_part2/', output_path='results_part2/')
    
    analyzer.create_param_boxplot(range(1,16), cmp=None, export_values=True)
    analyzer.create_param_boxplot(range(1,16), cmp='problem', export_values=True)
    analyzer.create_param_boxplot(range(1,16), cmp='dynamic', export_values=True)
    
    analyzer.create_param_scatter_matrix(range(1,16), cmp=None)
    analyzer.create_param_scatter_matrix(range(1,16), cmp='problem')
    analyzer.create_param_scatter_matrix(range(1,16), cmp='dynamic')
    
    analyzer.create_parameter_importance_csv(range(1,16))
    analyzer.create_param_importance_plot(range(1,16), cmp=None)
    analyzer.create_param_importance_plot(range(1,16), cmp='dynamic')
    analyzer.create_param_importance_plot(range(1,16), cmp='problem')

    for i in (range(1,16)):
        analyzer.create_partial_dep_plot(i, n_points=60)

    analyzer.create_best_param_csv(range(1,16))
    analyzer.create_categorical_analysis_csv(range(1,16))
    
    # get best param set
    for i in (range(1,16)):
        print(analyzer.get_best_param_cli(i))
    
