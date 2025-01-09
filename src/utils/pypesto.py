# def save_pypesto_results(result, filename):
    
#     assert(filename.endswith('.hdf5'))
    
#     if os.path.exists(f'/SBML/PyPESTO/FRP/Results') == False:
#         os.makedirs(f'/SBML/PyPESTO/FRP/Results')
#     if os.path.exists(f'/SBML/PyPESTO/FRP/Results/{MODEL_NAME}') == False:
#         os.makedirs(f'/SBML/PyPESTO/FRP/Results/{MODEL_NAME}')
    
#     result_file = f'/SBML/PyPESTO/FRP/Results/{MODEL_NAME}/{filename}'
    
#     pypesto.store.write_result(
#         result=result,
#         filename=result_file,
#         problem=True,
#         optimize=True,
#         profile=True,
#         sample=True,
#     )