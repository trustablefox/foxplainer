from __future__ import print_function

from .lrxp import LRExplainer
from .options import Options
from .rndmforest import XRF, Dataset
from .html_string import HtmlString

import pandas as pd
import ipywidgets as widgets
import collections
import matplotlib.pyplot as plt

class FoX(object):
    """A FoX object should be initialized with the following attributes to perform the logical explanation

    Parameters
    ----------
    global_model_name : :obj:`str` 
        The black-box global model to be explained, currently support 2 models, 'LR' for Logistic Regression and 'RF' for Random Forest
    xtype : :obj:`str` optional
        Explanation type, currently support 2 types, 'abd' for Abductive Explanation and 'con' for Concretive Explanation
    xnum : :obj:`int` or :obj:`str`
        Number of explanations to be generated for each instance, this should be either 1 or "all"
    global_model_path : :obj:`str`
        Path to the global model file in .pkl format trained by the sklearn library
    proj_name : :obj:`str`
        Project name
    data_path : :obj:`str`
        Path to the data files required for the FoX
    inst_id : :obj:`int`
        The row index of the instance to be explained
    """

    def __init__(self, global_model_name=None, xtype='abd', xnum=1, global_model_path=None, proj_name=None, data_path=None, inst_id=0):
        self.options = Options(global_model_name=global_model_name, 
                               xtype=xtype, 
                               xnum=xnum,
                               global_model_path=global_model_path,
                               proj_name=proj_name,
                               data_path=data_path)
        self.explainer = None
        self.inst_id = inst_id
        self.tab_nest = widgets.Tab()
        self.accordion = widgets.Accordion(children=[self.tab_nest])
        self.explained_instance = ""
        self.abd_con_exp_html = ""
        self.abd_exp_html = ""
        self.con_exp_html = ""
        self.ffa_exp_html = ""
        self.instance_info_html = ""

    def exp_to_html(self, exp_list, exp_type, explained_instance):
        for exp in exp_list:
            exp = self.exp_mapping(exp)
            if self.explained_instance == "":
                self.explained_instance = HtmlString(list_of_pair=explained_instance, exp_type=self.options.xtype, is_explained_instance=True).get_html()
            self.instance_info_html += self.explained_instance
            if exp_type == "abd":
                self.abd_exp_html += HtmlString(list_of_pair=exp, exp_type="abd").get_html()
            elif exp_type == "con":
                self.con_exp_html += HtmlString(list_of_pair=exp, exp_type="con").get_html()
            elif exp_type == "ffa":
                self.ffa_exp_html += HtmlString(list_of_pair=exp, exp_type="ffa").get_html()

    def show_in_jupyter(self, show_both_exp=False) -> None:
        if show_both_exp:
            self.accordion.set_title(index=0, title=[f"Instance ID {self.inst_id}"])
            abd_exp_html = widgets.HTML(value=self.abd_exp_html)
            con_exp_html = widgets.HTML(value=self.con_exp_html)
            if self.ffa_exp_html != "":
                ffa_exp_html = widgets.HTML(value=self.ffa_exp_html)
                self.tab_nest.children = [abd_exp_html, con_exp_html, ffa_exp_html] 
            else:
                #instance_info_html = widgets.HTML(value=self.instance_info_html)
                self.tab_nest.children = [abd_exp_html, con_exp_html] # add "instance_info_html" later
            self.tab_nest.set_title(index=0, title="Abductive Exp.")
            self.tab_nest.set_title(index=1, title="Contrastive Exp.")
            self.tab_nest.set_title(index=2, title="Explained Instance")
            self.tab_nest.set_title(index=3, title="FFA Exp.")
        else:
            self.accordion.set_title(index=0, title=[f"Instance ID {self.inst_id}"])
            abd_con_exp_html = widgets.HTML(value=self.abd_con_exp_html)
            #instance_info_html = widgets.HTML(value=self.instance_info_html)
            self.tab_nest.children = [abd_con_exp_html] # add "instance_info_html" later
            if self.options.xtype == "abd":
                exp_title = "Abductive Exp."
            elif self.options.xtype == "con":
                exp_title = "Contrastive Exp."
            elif self.options.xtype == "ffa":
                exp_title = "FFA (TODO)"
            self.tab_nest.set_title(index=0, title=exp_title)
            self.tab_nest.set_title(index=1, title="Explained Instance")
        from IPython.display import display
        return display(self.accordion)

    def explain(self, in_jupyter=False):
        """ Main function to perform the logical explanation
    
        """
        if in_jupyter:
            self.options.in_jupyter = True
        options = self.options
        # explaining
        if options.xtype:
            print('\nExplaining the {0} model...\n'.format('logistic regression' if options.global_model_name == 'LR' else 'random forest'))
            # Explain data
            data = Dataset(filename=options.data_path+options.proj_name+'.csv', mapfile=options.mapfile,
                        separator=options.separator, use_categorical=options.use_categorical)
            insts = pd.read_csv(options.data_path + options.proj_name + '_X_test.csv')
            for id in range(len(insts)):
                if id != self.inst_id:
                    continue
                inst = insts.iloc[id]
                # explain RF model
                if options.global_model_name == 'RF':
                    self.explainer = XRF(data, options)
                # explain LR model
                elif options.global_model_name == 'LR':
                    self.explainer = LRExplainer(data, options)
                
                _, _, explained_instance, explanation_list, explanation_size_list = self.explainer.explain(inst)
                if in_jupyter:
                    explained_instance = self.exp_mapping(explained_instance)
                    if self.options.xnum not in (-1, 'all'):
                        expl = explanation_list[self.options.xtype][0]
                        explanation = self.exp_mapping(expl)
                        if self.explained_instance == "":
                            self.explained_instance = HtmlString(list_of_pair=explained_instance,
                                                                 exp_type=self.options.xtype,
                                                                 is_explained_instance=True).get_html()
                        self.instance_info_html += self.explained_instance
                        self.abd_con_exp_html += HtmlString(list_of_pair=explanation,
                                                            exp_type=self.options.xtype).get_html()
                        self.show_in_jupyter()
                    else:
                        # i.e. enumeration
                        # abd exp
                        self.exp_to_html(exp_list=explanation_list['abd'], exp_type='abd', explained_instance=explained_instance)
                        self.exp_to_html(exp_list=explanation_list['con'], exp_type='con', explained_instance=explained_instance)
                        
                        ffa = self.ffa(explanation_list)
                        if ffa != {}:
                            ffa_explained_instance = []
                            for k, v in ffa.items():
                                ffa_explained_instance.append([k, v])
                            ffa_exp_list = "IF "
                            for kv in ffa_explained_instance:
                                ffa_exp_list += f"{kv[0]} = {kv[1]} AND "            
                            ffa_exp_list = ffa_exp_list.strip("AND ")
                            if len(explanation_list['abd']) > 0:
                                label = explanation_list['abd'][0].split("THEN")[-1].split("=")[0].strip()
                                pred = explanation_list['abd'][0].split("THEN")[-1].split("=")[1].strip()
                            else:
                                label = explanation_list['con'][0].split("THEN")[-1].split("=")[0].strip()
                                pred = explanation_list['con'][0].split("THEN")[-1].split("=")[1].strip()
                                pred = "True" if pred == "False" else "False"
                            ffa_exp_list += f" THEN {label} = {pred}"
                            ffa_exp_list = [ffa_exp_list]
                            self.exp_to_html(exp_list=ffa_exp_list, exp_type='ffa', explained_instance=ffa_explained_instance)
                        self.show_in_jupyter(show_both_exp=True)                        
                else:
                    if self.options.xnum not in (-1, 'all'):
                        exp_type_name = "Abductive" if self.options.xtype == "abd" else "Contrastive"
                        expl = explanation_list[self.options.xtype][0]
                        explanation_size = explanation_size_list[self.options.xtype][0]
                        print("Explained Instance\n", explained_instance, f"\n\n{exp_type_name} Explanation\n", expl, "\n\n", explanation_size, "\n")
                    else:
                        print("Explained Instance\n ", explained_instance)
                        for xtype in ['abd', 'con']:
                            exp_type_name = "Abductive" if xtype == "abd" else "Contrastive"
                            print(f"\n{exp_type_name} Explanation")
                            for i, expl in enumerate(explanation_list[xtype]):
                                explanation_size = explanation_size_list[xtype][i]
                                print(' ', expl, "\n\n ", explanation_size, "\n\n")
                        ffa = self.ffa(explanation_list)
                        print('FFA:\n{}'.format(ffa))

    def exp_mapping(self, if_else_text):
        # use list to preserve the order of the if-else statements
        mapped = []
        # map features
        feature_value = if_else_text.split('THEN')[0]
        feature_value = feature_value.split('AND')
        feature_value = [word.strip("IF ") for word in feature_value]
        for fea_val_pair in feature_value:
            fea_val = fea_val_pair.split('=')
            mapped.append([fea_val[0].strip(), round(float(fea_val[1].strip()), 5)])
        # map label
        label_value = if_else_text.split('THEN')[1].strip().split("=")
        mapped.append([label_value[0].strip(), label_value[1].strip()])
        return mapped
    
    def ffa(self, explanation_list):
        """
        unweighted feature attribution
        """
        axps = map(lambda l: l.split('IF ', maxsplit=1)[-1].rsplit(' THEN ', maxsplit=1)[0].split(' AND '), 
                   explanation_list['abd'])
        axps_ = []
        for xp in axps:
            axps_.append(list(map(lambda l: l.split(' = ', maxsplit=1)[0].strip(), xp)))

        lit_count = collections.defaultdict(lambda: 0)
        nof_axps = len(axps_)
        for axp in axps_:
            for lit in axp:
                lit_count[lit] += 1
        lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
        return lit_count

    def visulise_ffa(self, f2imprt):
        names = []
        values = []
        for f in sorted(f2imprt.keys(), key=lambda l: (abs(f2imprt[l]), l)):
            names.append(f)
            values.append(f2imprt[f])
        
        plt.rcParams['axes.linewidth'] = 2
        fig, ax = plt.subplots()
        # Fig size
        fig.set_size_inches(4, 4)

        for n, v in zip(names, values):
            if v > 0:
                ax.barh(y=[n], width=[v], alpha=0.4, height=0.3, color=(0.2, 0.4, 0.6, 0.6))  # '#86bf91', zorder=2)
            else:
                ax.barh(y=[n], width=[v], alpha=0.8, height=0.3, color='orange')  # 86bf91')

        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_visible(False)
        #ax.spines['bottom'].set_position('zero')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='y', pad=3, labelsize=15)

        for h, (n, v) in enumerate(zip(names, values)):
            ax.text(v, h+.18, '{:.2f}'.format(v), color='black',
                    horizontalalignment='left' if v > 0 else 'right',
                    #verticalalignment='top',
                    #(0.2, 0.4, 0.6, 0.6),
                    fontsize=15)#, fontweight='bold')

            ax.text(-.003 if v > 0 else .003, h-.05, n, color='black',
                    horizontalalignment='right' if v > 0 else 'left',
                    #verticalalignment='top',
                    #(0.2, 0.4, 0.6, 0.6),
                    fontsize=15)#, fontweight='bold')

        plt.show()
        plt.close()
