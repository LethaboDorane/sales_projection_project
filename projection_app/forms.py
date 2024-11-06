from django import forms

class SalesForm(forms.Form):
    sales_data = forms.CharField(label="Sales Data (comma-separated)", widget=forms.Textarea)
    months = forms.IntegerField(label="Projection Period")
    graph_type = forms.ChoiceField(choices=[('bar', 'Bar'), ('line', 'Line'), ('scatter', 'Scatter')])
    data_frequency = forms.ChoiceField(
        label="Data Frequency",
        choices=[('monthly', 'Monthly'), ('quarterly', 'Quarterly')],
        initial='quarterly'
    )
