from __future__ import print_function
from .lrxp import LRExplainer
from .options import Options
from .rndmforest import XRF, Dataset
from .html_string import HtmlString
import pandas as pd
import ipywidgets as widgets


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

    def show_in_jupyter(self, show_both_exp=False) -> None:
        if show_both_exp:
            self.accordion.set_title(index=0, title=[f"Instance ID {self.inst_id}"])
            abd_exp_html = widgets.HTML(value=self.abd_exp_html)
            con_exp_html = widgets.HTML(value=self.con_exp_html)
            #instance_info_html = widgets.HTML(value=self.instance_info_html)
            self.tab_nest.children = [abd_exp_html, con_exp_html] # add "instance_info_html" later
            self.tab_nest.set_title(index=0, title="Abductive Exp.")
            self.tab_nest.set_title(index=1, title="Contrastive Exp.")
            self.tab_nest.set_title(index=2, title="Explained Instance")
        else:
            self.accordion.set_title(index=0, title=[f"Instance ID {self.inst_id}"])
            abd_con_exp_html = widgets.HTML(value=self.abd_con_exp_html)
            #instance_info_html = widgets.HTML(value=self.instance_info_html)
            self.tab_nest.children = [abd_con_exp_html] # add "instance_info_html" later
            exp_title = "Abductive Exp." if self.options.xtype == "abd" else "Contrastive Exp."
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
                        self.show_in_jupyter(show_both_exp=True) 
                        # explanation_list['abd'] stores all AXps,  explanation_size_list['abd'] store all axp size
                        # explanation_list['con'] stores all CXps,  explanation_size_list['con'] store all cxp size
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
