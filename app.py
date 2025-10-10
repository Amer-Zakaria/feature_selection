import gradio as gr
from run_genetic_algo import run_genetic_algorithm


def process_csv(file="data.csv"):
    result = run_genetic_algorithm(file)

    if result["success"]:
        return "\n".join(result["messages"])
    else:
        return result["message"]


# Create Gradio interface
demo = gr.Interface(
    fn=process_csv,
    inputs=gr.Text(
        label="Data (CSV)",
    ),
    outputs=gr.Textbox(label="Results", lines=10),
    title="Genetic Algorithm Feature Selection",
    description="Upload a CSV file to run genetic algorithm feature selection",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
