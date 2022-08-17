class HtmlString(object):
    def __init__(self, list_of_pair, exp_type, is_explained_instance=False) -> None:
        self.list_of_pair = list_of_pair
        self.exp_type = exp_type
        self.is_explained_instance = is_explained_instance
        self.css = """
        .box{
        border: 1px solid black;
        border-radius: 20px;
        height: auto;
        width: auto;
        margin-top:10px;
        /* background-color: red; */
        }

        .box-wide{
        border: 1px solid black;
        border-radius: 20px;
        height: auto;
        width: auto;
        margin-top:10px;
        /* background-color: red; */
        }

        .inner-box {
        background-color: rgba(70,161,255,255);
        border-top-left-radius: 18px;
        border-top-right-radius: 18px;
        padding:4px;
        padding-left: 28px;
        }

        .title {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: white;
        font-size: 12pt;
        font-weight: 600;
        }

        .bot-box {
        padding:10px 12px;
        }

        .bot-box-two {
        padding:10px 12px;
        grid-template-rows: 1fr 1fr;
        }

        .bot-box-three {
        padding:10px 12px;
        display: grid;
        grid-template-rows: 1fr 1fr 1fr;
        }

        /* (800-100-80-20)/3 */
        .bot-container {
        display: grid;
        grid-template-columns: 15% 25% 5% 25% 5% 25%;
        margin-bottom: 6px;
        justify-content: center;
        align-items: center;
        }

        .general-button {
        border-radius: 10px;
        height:40px;
        width: 96px;
        background-color: rgba(70,161,255,255);
        justify-content: center;
        align-items: center;
        display: flex;
        /* margin-right: 15px; */
        }

        .general-text {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: white;
        font-size: 10pt;
        text-align: center;
        }

        .input-box {
        height:50px;
        width: 100%;
        justify-content: center;
        align-items: center;
        display: flex;
        }

        .input-inner-box {
        background-color: rgba(86,194,255,255);
        border-radius: 10px;
        height:40px;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0 5px;
        }

        .input-inner-box-grid {
        background-color: rgba(86,194,255,255);
        border-radius: 10px;
        height:40px;
        width: 100%;
        display: grid;
        grid-template-columns: 68% 32%;
        justify-content: center;
        align-items: center;
        padding: 0 5px;
        }

        .input-inner-text {
        justify-content: center;
        align-items: center;
        display: flex;
        font-size: 20px;

        }

        .inner-symbol {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: black;

        }

        .input-container {
        border-radius: 10px;
        height: 80%;
        width: 100%;
        background-color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        }

        .input_text {
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size:14px;
        color: black
        }
        """
        self.html_body = self.generate_html_body()
        self.html = self.generate_whole_html()
        
    
    def generate_html_body(self):
        list_of_pair = self.list_of_pair
        exp_type = self.exp_type
        label_value = [list_of_pair[-1][0], list_of_pair[-1][1]]
        del list_of_pair[-1]
        # check exp_type to decide whether to alter label value
        if self.exp_type == "con":
            label_value[1] = "True" if label_value[1] == "False" else "False"
        # prediction color
        if label_value[1] == "True":
            color = "rgba(237,34,14,255)"
        else:
            color = "rgba(96,217,55,255)"

        if exp_type == "abd":
            title = "Why?"
            exp_type_full = "Abductive Explanation"
        elif exp_type == "con":
            title = "How?"
            exp_type_full = "Contrastive Explanation"

        if self.is_explained_instance:
            title = "Explained Instance"
            exp_type_full = ""

        title_html = f'''
                        <div class="box">
                            <div class="inner-box">
                                <text class="title">{title} {exp_type_full}</text>
                            </div>
                        '''
        if_text = "If"
        if self.is_explained_instance:
            if_text = "Feature Values"
        if_box_html = f'''
                        <div class="bot-box-two">
                            <div class="bot-container">
                                <div class="general-button">
                                    <text class="general-text">{if_text}</text>
                                </div>
                        '''

        condition_html = self.generate_condition_html(fea_val_pair=list_of_pair)
        then_text = "Then"
        if self.is_explained_instance:
            then_text = "Prediction"
        then_box_html = f'''
                        </div>
                        <div class="bot-container">
                            <div class="general-button">
                                <text class="general-text">{then_text}</text>
                            </div>
                        '''
        equal_sign = "&nbsp;&nbsp;&nbsp;&nbsp;="
        if self.exp_type == "con":
            equal_sign = " could change to"
            label_value[0] = "Defect"
        class_box_html = f'''
                                    <div class="input-box">
                                        <div class="input-inner-box-grid" style="background-color: {color}">
                                            <text class="general-text">{label_value[0]}{equal_sign}</text>
                                            <div class="input-container">
                                                <text class="input_text">{label_value[1]}</text>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        '''
        return title_html + if_box_html + condition_html + then_box_html + class_box_html

    def generate_condition_html(self, fea_val_pair):
        con_html = ""
        max_len = len(fea_val_pair)
        i = 0
        for fea_name, val in fea_val_pair:
            equal_sign = "&nbsp;&nbsp;&nbsp;&nbsp;="
            # check if the val is int or float
            val = str(val).strip(".0")
            if "!" in fea_name:
                fea_name = fea_name.replace("!", "")
                equal_sign = "&nbsp;&nbsp;&nbsp;&nbsp;!="
            if val == "":
                val = "0"
            if i == max_len - 1:
                con_html += f'''
                    <div class="input-box">
                        <div class="input-inner-box-grid">
                            <text class="general-text">{fea_name}{equal_sign}</text>
                            <div class="input-container">
                                <text class="input_text">{val}</text>
                            </div>
                        </div>
                    </div>
                '''
            else:
                con_html += f'''
                                <div class="input-box">
                                    <div class="input-inner-box-grid">
                                        <text class="general-text">{fea_name}{equal_sign}</text>
                                        <div class="input-container">
                                            <text class="input_text">{val}</text>
                                        </div>
                                    </div>
                                </div>
                                <div class="input-inner-text">
                                    <text class="inner-symbol">&</text>
                                </div>
                            '''
            i += 1
        return con_html

    def generate_whole_html(self):
        css = self.css
            
        head = '''
            <!DOCTYPE HTML>
            <html>
            <head>
            <meta charset="utf-8">
            <title>template.html</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
        '''
        
        style = '''
            <style>
                %s
            </style>
        ''' %(css)

        
        body = f'''
                <body>
                <div id="container">
                    {self.html_body}
                </div>  
                </body>
                ''' 
        html = f'''
                {head}
                {style}
                {body}
                '''
        self.html = html
        return html
    
    def get_html(self):
        return self.html