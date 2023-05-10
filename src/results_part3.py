from analyzer import Analyzer


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':

    keyorder1 = ['eil51', 'rat195', 'berlin52', 'gil262',
                 'pr136', 'lin318', 'pr226', 'pr439', 'd198', 'fl417']
    keyorder2 = ['eil51', 'berlin52', 'pr136', 'pr226',
                 'd198', 'rat195', 'gil262', 'lin318', 'pr439', 'fl417']

    analyzer = Analyzer(
        mode='exp', results_path='output_part3/output_part3_Reference/', output_path='results_part3/')
    

    analyzer.create_reset_accuracy_csv(
        range(1, 31), all_stats=True, custom_keyorder=keyorder1)
    analyzer.create_reset_accuracy_csv(
        range(1, 31), all_stats=False, custom_keyorder=keyorder1)

    analyzer.create_pr_plot(range(1, 31), cmp=None, grouped=True)
    analyzer.create_pr_plot(range(1, 31), cmp='dynamic', grouped=False)
    analyzer.create_pr_plot(range(1, 31), cmp='problem',
                            grouped=False, custom_keyorder=keyorder1)

    analyzer.create_run_plot(range(1, 31), cmp='dynamic', aggr='problem', iteration_range=[
                             1990, 2299], custom_keyorder=keyorder1)
    analyzer.create_run_plot(range(1, 31), y_value='swaps',
                             cmp='dynamic', aggr=None, y_value_range=[0, 5.5], iteration_range=[1990, 2299])

    analyzer = Analyzer(
        mode='exp', results_path='output_part3/output_part3_HPO/', output_path='results_part3/')

    analyzer.create_reset_accuracy_csv(
        range(1, 31), all_stats=False, custom_keyorder=keyorder1)

    analyzer.create_pr_plot(range(1, 31), cmp=None, grouped=True)
    analyzer.create_pr_plot(range(1, 31), cmp='dynamic', grouped=False)
    analyzer.create_pr_plot(range(1, 31), cmp='problem',
                            grouped=False, custom_keyorder=keyorder1)
    analyzer.create_pr_plot(range(1, 31), cmp='detection_threshold')
    analyzer.create_fp_theta_plot(range(1, 31))

    analyzer.create_run_plot(range(1, 31), cmp='dynamic', aggr=None)
    analyzer.create_run_plot(range(1, 31), cmp='dynamic', aggr='problem', iteration_range=[
                             1990, 2299], custom_keyorder=keyorder1)

    analyzer.create_run_plot(range(1, 31), y_value='swaps',
                             cmp='dynamic', aggr=None, y_value_range=[0, 5.5], iteration_range=[1990, 2299])
    analyzer.create_run_plot(range(1, 31), y_value='swaps', cmp='dynamic', aggr='problem', y_value_range=[
                             0, 6], iteration_range=[1990, 2299], custom_keyorder=keyorder1)

    analyzer.create_run_plot(range(1, 31), cmp='folder', aggr=None,
                             alt_results_path='output_part3/output_part3_Reference/', y_value_range=[0.05, 0.45])
    analyzer.create_run_plot(range(1, 31), cmp='folder', aggr='dynamic', iteration_range=[1990, 2299],
                             alt_results_path='output_part3/output_part3_Reference/')
