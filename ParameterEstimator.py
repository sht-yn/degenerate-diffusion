from project_imports import (
    np, optimize, # scipy.optimize は optimize としてインポート
    pm, pytensor, pt, as_op, az,
    Optional, Callable, List, Tuple, Dict, # Typing
    widgets, interact, fixed, FloatSlider, IntSlider, FloatText, IntText, # ipywidgets
    logging
)

class ParameterEstimator:
    """
    与えられた目的関数（例：対数疑似尤度）に基づいて、
    M推定、ベイズ推定、ワンステップ推定を実行する汎用的な関数群を提供するクラス。
    このクラスのメソッドはすべて静的メソッドとして呼び出せます。
    """

    @staticmethod
    def m_estimate(
        objective_function: Callable[[np.ndarray], float], 
        search_bounds: List[Tuple[Optional[float], Optional[float]]], 
        initial_guess: np.ndarray, 
        method: str = "L-BFGS-B", 
        **optimizer_options
    ) -> np.ndarray:
        """
        M推定（最尤推定または疑似最尤推定に相当）を実行します。
        objective_function を最小化するパラメータを見つけます。

        Args:
            objective_function (callable): 最小化を目指す目的関数。
                                           パラメータのNumPy配列を引数に取り、スカラー値を返す。
                                           例: `lambda params: my_negative_log_likelihood(params, data)`
            search_bounds (list[tuple]): パラメータの探索範囲。各要素は (min, max) のタプル。
                                         例: [(0.1, 10.0), (-5.0, 5.0), (None, None)] (Noneは境界なし)
            initial_guess (np.ndarray): パラメータの初期推定値。
            method (str, optional): scipy.optimize.minimize に渡す最適化手法。
            **optimizer_options: scipy.optimize.minimize に渡す追加のオプション。

        Returns:
            scipy.optimize.OptimizeResult: 最適化結果。
        """
        num_params = len(search_bounds)
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")
        if not isinstance(search_bounds, list) or \
           not all(isinstance(b, tuple) and len(b) == 2 for b in search_bounds):
            raise ValueError("search_bounds must be a list of (min, max) tuples.")
        # None を +/- np.inf に変換 (scipy.optimize.minimizeのbounds仕様に合わせる)
        processed_bounds = []
        for low, high in search_bounds:
            processed_bounds.append((
                -np.inf if low is None else low,
                np.inf if high is None else high
            ))
            if (low is not None and high is not None) and not (low <= high):
                 raise ValueError(f"In search_bounds, lower bound {low} cannot be greater than upper bound {high}.")
        
        if not isinstance(initial_guess, np.ndarray) or initial_guess.shape != (num_params,):
            raise ValueError(f"initial_guess must be a numpy array of shape ({num_params},)")

        # print(f"Starting M-estimation with initial guess: {initial_guess}, method: {method}")
        
        objective = lambda x: -objective_function(x) # ラッパー関数を作成

        # objective_function は最小化すべき関数そのものと仮定
        result = optimize.minimize(
            objective, 
            x0=initial_guess,
            method=method,
            bounds=processed_bounds, # 処理済みの境界を使用
            **optimizer_options
        )
        if result.success:
            # print(f"M-estimation successful. Estimated parameters: {result.x}")
            # print(f"Function value at minimum: {result.fun}") # ユーザーが提供した objective_function の最小値
            pass
        else:
            print(f"M-estimation FAILED: {result.message}")
        return result.x

    @staticmethod
    def bayes_estimate(
        objective_function: Callable[[np.ndarray], float], # 擬似対数尤度を想定
        search_bounds: List[Tuple[Optional[float], Optional[float]]],
        initial_guess: np.ndarray,
        prior_log_pdf: Optional[Callable[[np.ndarray], float]] = None,
        method: str = "mcmc",
        draws: int = 2000, tune: int = 500, chains: int = 4, cores: int = 4,
        map_optimizer_method: str = "L-BFGS-B",
        **options
    ) -> np.ndarray:
        """
        ベイズ推定を実行します。objective_function は対数尤度を返すものとします。
        """
        num_params = len(search_bounds)
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")
        if prior_log_pdf is not None and not callable(prior_log_pdf):
            raise TypeError("If provided, prior_log_pdf must be a callable function.")
        if not isinstance(search_bounds, list) or \
           not all(isinstance(b, tuple) and len(b) == 2 for b in search_bounds):
            raise ValueError("search_bounds must be a list of (min, max) tuples.")
        
        processed_bounds = [] # MCMCのデフォルトUniform Prior用とMAPの境界用
        for low, high in search_bounds:
            processed_bounds.append((
                -np.inf if low is None else low,
                np.inf if high is None else high
            ))
            if (low is not None and high is not None) and not (low <= high):
                 raise ValueError(f"In search_bounds, lower bound {low} cannot be greater than upper bound {high}.")

        if not isinstance(initial_guess, np.ndarray) or initial_guess.shape != (num_params,):
            raise ValueError(f"initial_guess must be a numpy array of shape ({num_params},)")

        param_dtype = pytensor.config.floatX
        vector_pytensor_type = pytensor.tensor.TensorType(dtype=param_dtype, shape=(None,)) # 1Dベクトル
        scalar_pytensor_type = pytensor.tensor.TensorType(dtype=param_dtype, shape=())      #

        if method.lower() == "mcmc":
            # print(f"Starting MCMC estimation with PyMC...")
            # print(f"  draws={draws}, tune={tune}, chains={chains}, cores={cores}")

            @as_op(itypes=[vector_pytensor_type], 
                      otypes=[scalar_pytensor_type])
            def log_likelihood_op(theta_numpy: np.ndarray) -> np.ndarray:
                val = objective_function(theta_numpy) # objective_function が -logL を返す
                return np.asarray(val, dtype=param_dtype)

            wrapped_user_prior_log_pdf_op = None
            if prior_log_pdf is not None:
                @as_op(itypes=[pt.vector(dtype=param_dtype)], 
                          otypes=[pt.scalar(dtype=param_dtype)])
                def temp_prior_wrapper(theta_numpy: np.ndarray) -> np.ndarray:
                    val = prior_log_pdf(theta_numpy)
                    return np.asarray(val, dtype=param_dtype)
                wrapped_user_prior_log_pdf_op = temp_prior_wrapper
            
            with pm.Model() as pymc_model:
                theta_rv_components = []
                param_names_in_model = []
                for i in range(num_params):
                    param_name = f'param_{i}'
                    param_names_in_model.append(param_name)
                    low_b, high_b = processed_bounds[i] # 処理済みの境界を使用

                    # print(f"Info: Parameter '{param_name}' using pm.Uniform with bounds ({low_b}, {high_b}).")
                    # if np.isinf(low_b) or np.isinf(high_b):
                    #     if not (np.isinf(low_b) and np.isinf(high_b)):
                    #          print(f"  Warning: This will be an improper prior for '{param_name}'. Ensure posterior is proper.")
                    theta_rv_components.append(pm.Uniform(param_name, lower=low_b, upper=high_b))
                
                theta_tensor = pt.stack(theta_rv_components)
                
                log_likelihood_term = log_likelihood_op(theta_tensor) 
                pm.Potential('custom_total_likelihood', log_likelihood_term)

                if wrapped_user_prior_log_pdf_op:
                    user_prior_term = wrapped_user_prior_log_pdf_op(theta_tensor)
                    pm.Potential('user_specified_prior', user_prior_term)
                
                idata = None
                posterior_mean_theta = np.full(num_params, np.nan, dtype=float)

                pymc_logger = logging.getLogger("pymc")
                original_level = pymc_logger.level
                pymc_logger.setLevel(logging.WARNING)
                
                try:
                    initvals_list = []
                    if num_params > 0 and chains > 0:
                        for _ in range(chains):
                            chain_init_dict = {}
                            noise = np.random.normal(scale=np.abs(initial_guess)*0.01 + 1e-5, size=num_params) # 各チェーンでノイズを変える
                            perturbed_initial_guess = initial_guess + noise
                            for i in range(num_params):
                                param_name = param_names_in_model[i]
                                val = perturbed_initial_guess[i]
                                low_b_init, high_b_init = processed_bounds[i]
                                if np.isfinite(low_b_init) and np.isfinite(high_b_init) and low_b_init < high_b_init:
                                    margin = (high_b_init - low_b_init) * 1e-3
                                    if margin == 0: margin = 1e-6
                                    val = np.clip(val, low_b_init + margin, high_b_init - margin)
                                elif np.isfinite(low_b_init): val = max(val, low_b_init + 1e-3 if high_b_init > low_b_init + 1e-3 else low_b_init)
                                elif np.isfinite(high_b_init): val = min(val, high_b_init - 1e-3 if low_b_init < high_b_init - 1e-3 else high_b_init)
                                chain_init_dict[param_name] = val
                            initvals_list.append(chain_init_dict)
                    

                    idata = pm.sample(
                        draws=draws, tune=tune, chains=chains, cores=cores, step = pm.Metropolis(),
                        initvals=initvals_list if initvals_list else None,
                        progressbar= False
                    )
                    # print("MCMC sampling completed.")
                    if idata is not None and num_params > 0:
                        means_list = []
                        for i in range(num_params):
                            param_name = param_names_in_model[i]
                            if param_name in idata.posterior:
                                param_mean = idata.posterior[param_name].mean(dim=("chain", "draw")).values
                                means_list.append(param_mean.item() if param_mean.ndim == 0 else param_mean)
                            else: means_list.append(np.nan)
                        posterior_mean_theta = np.array(means_list, dtype=float)
                        # print(f"  Posterior mean of theta: {posterior_mean_theta}")
                except Exception as e:
                    print(f"Error during PyMC sampling: {e}")
                return posterior_mean_theta

        # elif method.lower() == "map":
        #     effective_prior_log_pdf: callable
        #     if prior_log_pdf is None:
        #         def default_uniform_prior_log_pdf(params_check: np.ndarray) -> float:
        #             for i_param, p_val in enumerate(params_check):
        #                 low, high = processed_bounds[i_param] # 処理済み境界を使用
        #                 if np.isinf(low) and np.isinf(high): continue
        #                 if not (low <= p_val <= high): return -np.inf
        #             return 0.0
        #         effective_prior_log_pdf = default_uniform_prior_log_pdf
        #     else:
        #         effective_prior_log_pdf = prior_log_pdf

        #     def negative_log_posterior(params):
        #         for i, (low, high) in enumerate(processed_bounds): # 処理済み境界を使用
        #             if not (low <= params[i] <= high): return np.inf
        #         log_prior_val = effective_prior_log_pdf(params)
        #         if not np.isfinite(log_prior_val): return np.inf
        #         objective_val = objective_function(params)
        #         if not np.isfinite(objective_val): return np.inf
        #         return objective_val - log_prior_val

        #     print(f"Starting MAP estimation with initial guess: {initial_guess}, method: {map_optimizer_method}")
        #     map_optimizer_options = options.get('optimizer_options', {}).copy()
        #     if 'bounds' not in map_optimizer_options and map_optimizer_method in ["L-BFGS-B", "TNC", "SLSQP"]:
        #          map_optimizer_options['bounds'] = processed_bounds # 処理済みの境界を使用
            
        #     map_result = optimize.minimize(
        #         negative_log_posterior, x0=initial_guess, method=map_optimizer_method, **map_optimizer_options
        #     )
        #     if map_result.success: print(f"MAP estimation successful. Estimated parameters: {map_result.x}")
        #     else: print(f"MAP estimation FAILED: {map_result.message}")
        #     return {"map_estimate": map_result.x if map_result.success else None, "optimize_result": map_result}
        # else:
        #     raise ValueError(f"Unknown Bayesian estimation method: {method}. Supported: 'map', 'mcmc'.")

    @staticmethod
    def one_step_estimate(
        objective_function: Callable[[np.ndarray], float],
        search_bounds: List[Tuple[Optional[float], Optional[float]]],
        initial_estimator: np.ndarray,
        gradient_objective_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hessian_objective_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_numerical_derivatives: bool = True
    ) -> np.ndarray:
        num_params = len(search_bounds)
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")
        if not isinstance(search_bounds, list) or \
           not all(isinstance(b, tuple) and len(b) == 2 for b in search_bounds):
            raise ValueError("search_bounds must be a list of (min, max) tuples.")
        
        processed_bounds = [] # 処理済みの境界（クリッピング用）

        for low, high in search_bounds:
            processed_bounds.append((
                -np.inf if low is None else low,
                np.inf if high is None else high
            ))
            if (low is not None and high is not None) and not (low <= high):
                 raise ValueError(f"In search_bounds, lower bound {low} cannot be greater than upper bound {high}.")

        if not isinstance(initial_estimator, np.ndarray) or initial_estimator.shape != (num_params,):
            raise ValueError(f"initial_estimator must be a numpy array of shape ({num_params},)")

        # print(f"Starting One-Step estimation from: {initial_estimator}")
        grad_approx = None
        hess_inv_approx = None

                # print(f"  One-Step updated parameters: {theta_new}")
        def get_scalar(value):
            if isinstance(value, np.ndarray):
                if value.size == 1:  # 要素数が1であることも確認
                    return value.item()
                else:
                    raise ValueError("Input NumPy array must have a single element to be converted to a scalar.")
            elif isinstance(value, (float, int)): # Pythonのネイティブな数値型
                return float(value) # floatに統一するなら
            else:
                raise TypeError(f"Unsupported type: {type(value)}. Expected float, int, or a single-element NumPy array.")

        if callable(gradient_objective_function) and not use_numerical_derivatives:
            grad_approx = gradient_objective_function(initial_estimator)
        else:
            epsilon = np.sqrt(np.finfo(float).eps)
            grad_approx = np.zeros_like(initial_estimator)
            for i in range(num_params):
                params_plus_eps = initial_estimator.copy()
                params_plus_eps[i] += epsilon
                params_minus_eps = initial_estimator.copy()
                params_minus_eps[i] -= epsilon
                f_plus = objective_function(params_plus_eps)
                f_minus = objective_function(params_minus_eps)
                f_plus = get_scalar(f_plus)
                f_minus = get_scalar(f_minus)
                grad_approx[i] = (f_plus - f_minus) / (2 * epsilon)
            # print(f"  Numerically approximated gradient: {grad_approx}")

        if callable(hessian_objective_function) and not use_numerical_derivatives:
            hess_matrix = hessian_objective_function(initial_estimator)
            try:
                hess_inv_approx = np.linalg.inv(hess_matrix)
            except np.linalg.LinAlgError:
                print("  Warning: Hessian matrix is singular. Using pseudo-inverse.")
            hess_inv_approx = np.linalg.pinv(hess_matrix)
        else:
            # 数値的にHessianを計算
            epsilon = np.sqrt(np.finfo(float).eps)
            hess_matrix = np.zeros((num_params, num_params))
            for i in range(num_params):
                for j in range(num_params):
                    params_pp = initial_estimator.copy()
                    params_pm = initial_estimator.copy()
                    params_mp = initial_estimator.copy()
                    params_mm = initial_estimator.copy()
                    params_pp[i] += epsilon
                    params_pp[j] += epsilon
                    params_pm[i] += epsilon
                    params_pm[j] -= epsilon
                    params_mp[i] -= epsilon
                    params_mp[j] += epsilon
                    params_mm[i] -= epsilon
                    params_mm[j] -= epsilon
                    f_pp = objective_function(params_pp)
                    f_pm = objective_function(params_pm)
                    f_mp = objective_function(params_mp)
                    f_mm = objective_function(params_mm)
                    f_pp = get_scalar(f_pp)
                    f_pm = get_scalar(f_pm)
                    f_mp = get_scalar(f_mp)
                    f_mm = get_scalar(f_mm)
                    hess_matrix[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
            try:
                hess_inv_approx = np.linalg.inv(hess_matrix)
            except np.linalg.LinAlgError:
                print("  Warning: Numerically computed Hessian is singular. Using pseudo-inverse.")
            hess_inv_approx = np.linalg.pinv(hess_matrix)
        
        one_step_update = np.dot(hess_inv_approx, grad_approx)
        theta_new = initial_estimator - one_step_update
        
        for i in range(num_params): # 更新後の値を境界内にクリップ
            theta_new[i] = np.clip(theta_new[i], processed_bounds[i][0], processed_bounds[i][1])


        return theta_new