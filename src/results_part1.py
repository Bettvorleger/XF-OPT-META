from analyzer import Analyzer
import pandas as pd


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':

    analyzer = Analyzer(
        mode='opt', results_path='output_part1/', output_path='results_part1/')

    analyzer.create_func_opt_boxplot(range(1,21))
    analyzer.create_convergence_plot(range(1,21))

    for ch in chunks(range(1, 21), 4):
        analyzer.create_convergence_plot(list(ch))

    writer_conover_t = pd.ExcelWriter('conover_t.xlsx', engine='openpyxl')
    writer_conover_p = pd.ExcelWriter('conover_p.xlsx', engine='openpyxl')
    writer_stats = pd.ExcelWriter('stats.xlsx', engine='openpyxl')
    kwh = {}

    for ch in chunks(range(1, 21), 4):
        stats, tests = analyzer.get_convergence_stats(list(ch))
        problem = stats.pop('problem')

        pd.DataFrame(stats, index=['auc', 'min']).to_excel(writer_stats, sheet_name=problem)

        kwh[problem] = {'pvalue':tests['kwh']['pvalue']}

        pd.DataFrame.from_dict(tests['conover_t']).to_excel(writer_conover_t, sheet_name=problem)
        pd.DataFrame.from_dict(tests['conover_p']).to_excel(writer_conover_p, sheet_name=problem)

    writer_conover_t.close()
    writer_conover_p.close()

    pd.DataFrame.from_dict(kwh).to_excel('kruskal.xlsx')

    stats, tests = analyzer.get_convergence_stats(range(1, 21), avg_res=True)
    pd.DataFrame(stats, index=['auc', 'min']).to_excel(writer_stats, sheet_name='mean')
    writer_stats.close()