import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64
from .forms import SalesForm
from django.http import HttpResponse
import os
from django.conf import settings

# Function to simulate seasonality effect with an offset for starting month
def add_seasonality(sales, months, amplitude=0.1, period=12, start_month=1):
    seasonal_effect = amplitude * np.sin(2 * np.pi * (np.arange(len(sales) + months) + (start_month - 1)) / period)
    return sales + seasonal_effect[:len(sales)], seasonal_effect[len(sales):]

# Function to add random noise to projections for more realism
def add_random_noise(future_sales, noise_level=0.05):
    noise = np.random.normal(0, noise_level * np.mean(future_sales), len(future_sales))
    return future_sales + noise

# Main view for handling the sales projection
def sales_projection_view(request):
    if request.method == 'POST':
        form = SalesForm(request.POST)
        if form.is_valid():
            sales_data = form.cleaned_data['sales_data']
            frequency = form.cleaned_data['data_frequency']  # Monthly or Quarterly
            periods = int(form.cleaned_data['months'])

            # Convert sales data to array, ensuring valid entries
            try:
                sales = np.array([float(x) for x in sales_data.split(',') if x.strip()])
            except ValueError:
                return render(request, 'projection_app/sales_projection.html', {
                    'form': form,
                    'error': 'Sales data should contain only numeric values separated by commas.'
                })

            # Handle quarterly data conversion if needed
            if frequency == 'quarterly':
                if len(sales) % 3 != 0:
                    return render(request, 'projection_app/sales_projection.html', {
                        'form': form,
                        'error': 'Quarterly data should be in multiples of 3 months (e.g., 3, 6, 9, etc.).'
                    })
                # Reshape data to calculate quarterly averages
                sales = sales.reshape(-1, 3).mean(axis=1)  # Average each set of 3 months into quarters
                projection_periods = periods // 3  # Convert monthly periods to quarterly
                period_label = "Quarter"
                seasonal_period = 4
            else:
                projection_periods = periods
                period_label = "Month"
                seasonal_period = 12

            if len(sales) == 0:
                return render(request, 'projection_app/sales_projection.html', {
                    'form': form,
                    'error': 'No valid sales data provided.'
                })

            # Proceed with model fitting and projections for quarterly data
            X = np.arange(1, len(sales) + 1).reshape(-1, 1)
            y = sales.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            # Generate future periods for the projection
            future_X = np.arange(len(sales) + 1, len(sales) + projection_periods + 1).reshape(-1, 1)
            future_sales = model.predict(future_X).flatten()

            # Add seasonality to both historical sales and future projections
            sales_with_seasonality, future_seasonal_effect = add_seasonality(sales, projection_periods, period=seasonal_period)
            future_sales_with_seasonality = future_sales + future_seasonal_effect

            # Add random noise to the future sales
            future_sales_noisy = add_random_noise(future_sales_with_seasonality)

            # Determine trend direction
            slope = model.coef_[0][0]
            if slope > 0:
                trend = "Positive (improving)"
            elif slope < 0:
                trend = "Negative (declining)"
            else:
                trend = "Stagnant"

            # Generate plot
            fig, ax = plt.subplots()
            graph_type = form.cleaned_data['graph_type']
            create_sales_graph(ax, graph_type, sales_with_seasonality, future_sales_noisy, X, future_X, projection_periods, period_label)

            # Save plot to a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()

            # Encode the image in base64
            graph = base64.b64encode(image_png).decode('utf-8')

            # Prepare future sales data for display
            future_sales_list = [(f"{period_label} {i}", f"{sale:.2f}") for i, sale in enumerate(future_sales_noisy, start=len(sales) + 1)]

            return render(request, 'projection_app/sales_projection.html', {
                'form': form,
                'graph': graph,
                'future_sales': future_sales_noisy,
                'future_sales_list': future_sales_list,
                'trend': trend,
            })
        
    else:
        form = SalesForm()

    return render(request, 'projection_app/sales_projection.html', {'form': form})

# Function to handle the graph type and plot creation
def create_sales_graph(ax, graph_type, sales_with_seasonality, future_sales_noisy, X, future_X, projection_periods, period_label):
    """Create the appropriate plot based on the selected graph type."""
    if len(sales_with_seasonality) != len(X.flatten()) or len(future_sales_noisy) != len(future_X.flatten()):
        raise ValueError("Mismatch in data length between actual and future sales for the chosen frequency.")

    if graph_type == 'bar':
        ax.bar(range(1, len(X) + 1), sales_with_seasonality, label='Actual Sales (with seasonality)', color='blue')
        ax.bar(range(len(X) + 1, len(X) + projection_periods + 1), future_sales_noisy, label=f'{projection_periods}-{period_label} Projection', color='red')
    elif graph_type == 'line':
        ax.plot(X, sales_with_seasonality, label='Actual Sales (with seasonality)', marker='o')
        ax.plot(future_X, future_sales_noisy, label=f'{projection_periods}-{period_label} Projection', linestyle='--', marker='x', color='red')
    elif graph_type == 'scatter':
        ax.scatter(X, sales_with_seasonality, label='Actual Sales (with seasonality)', color='blue')
        ax.scatter(future_X, future_sales_noisy, label=f'{projection_periods}-{period_label} Projection', color='red')
    
    ax.set(title="Sales and Future Projections", xlabel=period_label, ylabel="Sales")
    ax.legend()


# View to download the generated image
def download_image(request):
    """Serve the generated sales projection image as a download."""
    file_path = 'path_to_your_generated_image.png'  # Replace with the actual path
    file_format = 'image/png'  # Adjust based on the user's selection (png or jpg)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type=file_format)
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    else:
        return HttpResponse("File not found.", status=404)

# Function to generate and save the sales projection graph as an image file
def generate_sales_projection(data, file_format='png'):
    """Generate the sales projection plot and save it to a file."""
    # Create the plot
    fig, ax = plt.subplots()
    # Plot data (assuming `data` has the necessary information)
    # Your plotting code here...

    # Save the plot to the MEDIA folder for easy access
    file_path = os.path.join(settings.MEDIA_ROOT, f'sales_projection.{file_format}')
    plt.savefig(file_path, format=file_format)

    return file_path
