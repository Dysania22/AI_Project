from flask import Flask, request, render_template_string
from functions import scraper, find_string_with_numerals, make_clickable

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    query = ''
    df_html = ''
    if request.method == 'POST':
        query = request.form.get('query')
        df = scraper(query)
        # Convert the DataFrame to HTML with clickable links
        df_html = df.to_html(classes='data', header="true", escape=False, index=False, formatters={
                'Link': lambda x: f'<a href="{x}" target="_blank">{x}</a>'
        })

        return render_template_string(HTML_TEMPLATE, tables=[df_html])
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Patent Search Interface</title>
        <style>
            h1, h2 {
                text-align: center;
                font-size: 2em; /* Increase font size by 4 units (assuming 1em = 16px, this is 32px) */
            }
            .dataframe th {
                text-align: center;
                font-size: 1.25em; /* Increase font size by 4 units (assuming 1em = 16px, this is 20px) */
            }
            .dataframe td {
                white-space: nowrap; /* Prevent text from wrapping in table cells */
            }
            footer {
                text-align: center;
                font-size: 0.8em; /* Smaller font size */
                color: grey; /* Grey color */
                margin-top: 50px; /* Space above the footer */
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var textarea = document.getElementById('query');
                textarea.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault(); // Prevent newline in textarea
                        this.form.submit(); // Submit the form
                    }
                });
            });
        </script>
    </head>
    <body>
        <h1>Patent Search Interface</h1>
        <form method="post">
            <label for="query">Keyword Search:</label><br><br>
            <textarea id="query" name="query" rows="2" cols="25">{{ query }}</textarea><br><br>
            <input type="submit" value="Search">
       </form>
    {% if tables %}
      <h2>Results</h2>
      {% for table in tables %}
        {{ table|safe }}
      {% endfor %}
    {% endif %}
    <footer>
        Caleb Estes, Alan Crisologo, Corey Luksch, Alexandra Batko
    </footer>
  </body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)