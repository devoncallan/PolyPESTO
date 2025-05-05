import streamlit as st
import matplotlib.pyplot as plt

from polypesto.core.study import Study
from polypesto.models.CRP2 import IrreversibleCPE
from polypesto.visualization import (
    plot_comparisons_1D,
    plot_parameter_traces,
    plot_confidence_intervals,
    plot_sampling_scatter,
    plot_optimization_scatter
)


@st.cache_data
def load_study():
    return Study.load(
        "/PolyPESTO/experiments/irreversible_cpe/data/single_rxn", IrreversibleCPE
    )


# st.set_page_config(layout="wide")


study = load_study()
c_title, c1 = st.columns([2, 1])
c_title.title("Irreversible CPE Study")
c_title.write(
    "This is a Streamlit app to visualize the results of the Irreversible CPE study."
)


# c1, c2 = st.columns([1, 2])
# c1 = st.container()

condition_ids = study.get_condition_ids()
param_ids = study.get_parameter_ids()
# param_values = list(study.experiments.values())[0].true_params
# true_params= list(study.experiments.values())[0].true_params.to_dict()
# exps = [
#     list(study.get_experiments(filter_p_id=param_id).values())[0]
#     for param_id in param_ids
# ]
# param_values = [exp.true_params.to_dict() for exp in exps]


# # def get_param_value(param_id):
def get_param_value(param_id):
    params = study.simulation_params.by_id(param_id).to_dict()
    param_values = [f"{k}: {v}" for k, v in params.items()]
    param_value = ", ".join(param_values)
    return param_value

# st.write(study.simulation_params.to_dict())
# st.write("Parameter values:", param_values)
# st.write("Parameter IDs:", param_ids)
#     param_value = exp.true_params.to_dict()
# param_values_list = param_values.keys(
# param_values = [list(exp.true_params.to_dict()) for exp in study.experiments.values()]
c1l, c1r = c1.columns([1, 1])
selected_param_id = c1l.selectbox("Select a parameter ID", param_ids, format_func=get_param_value)
c1r.markdown(
    f"### {get_param_value(selected_param_id)}"
)  # Display the parameter values for the selected parameter ID
# selected_param_id =

experiments = study.get_experiments(filter_p_id=selected_param_id)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
axes = plot_comparisons_1D(study, selected_param_id, axes)

for ax in axes:
    ax.set_ylim(-2, 2)
#     ax.set_yscale("log")
c1.pyplot(fig, use_container_width=False)

# c2, _c = st.columns([2, 1])
c2 = st.container()
selected_cond_id = c_title.selectbox("Select a condition ID", condition_ids)
# selected_exp = study.get_experiments(
#     filter_cond_id=selected_cond_id, filter_p_id=selected_param_id
# )
selected_exp = study.get_experiments(
    filter_cond_id=selected_cond_id, filter_p_id=selected_param_id
)
result = study.results[(selected_cond_id, selected_param_id)]
true_params = list(selected_exp.values())[0].true_params.to_dict()
# c1.write(true_params)
# c1.write(
#     f"True parameters: {true_params}"
# )  # Display the true parameters for the selected experiment

# c2L, c2R = c2.columns([1, 1])
c21, c22, c23, c24 = c2.columns([1, 1, 1, 1])

c21.markdown("### Parameter traces")
fig, ax = plot_parameter_traces(result, true_params)
c21.pyplot(fig)

c22.markdown("### Confidence intervals")
fig, ax = plot_confidence_intervals(result, true_params)
c22.pyplot(fig)

c23.markdown("### Sampling scatter")
fig, ax = plot_sampling_scatter(result, true_params)
c23.pyplot(fig)

c24.markdown("### Optimization scatter")
fig, ax = plot_optimization_scatter(result, true_params)
c24.pyplot(fig)
 