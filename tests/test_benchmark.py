    "n_neurons": 120,
    "hidden_size": 64,
    "output_size": 32,
    "model_name": "offset10-model",
    "conditional": "time_delta",
    "num_steps": 5000,
    "batch_size": 512,
    "time_offset": 10,
lower_bound_score = {"r2_score": 30, "median_error": 1.5}
    "data_name": "rat-hippocampus-single-achilles",
    "dataset_initfunc": cebra.data.TensorDataset,
    "loader_initfunc": cebra.data.ContinuousDataLoader,
    "solver_initfunc": cebra.solver.SingleSessionSolver,
    return (
        neural_train.numpy(),
        neural_test.numpy(),
        label_train.numpy(),
        label_test.numpy(),
    )
    emb_train = solver.transform(train_set[torch.arange(
        len(train_set))]).numpy()
def _train(train_set, loader_initfunc, solver_initfunc):
    model = cebra.models.init(
        model_params["model_name"],
        model_params["n_neurons"],
        model_params["hidden_size"],
        model_params["output_size"],
    )
def _run(data_name, dataset_initfunc, loader_initfunc, solver_initfunc):
    neural_train, neural_test, label_train, label_test = _split_data(
        dataset, test_ratio)
    offset = cebra.data.datatypes.Offset(loader_params["time_offset"] // 2,
                                         loader_params["time_offset"] // 2)
    assert lower_bound_score["r2_score"] < r2_score
    assert lower_bound_score["median_error"] > pos_err
    print(f"{solver_initfunc.__name__}: r2 score = {r2_score:.4f}, "
          f"median abs error = {pos_err:.4f}")
