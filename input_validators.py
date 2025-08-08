class InputValidationError(Exception):
    pass

def validate_portfolio(portfolio):
    if not isinstance(portfolio, dict):
        raise InputValidationError("Portfolio must be a dictionary")
    if not portfolio:
        raise InputValidationError("Portfolio cannot be empty")
    for asset, weight in portfolio.items():
        if not isinstance(asset, str):
            raise InputValidationError("Asset names must be strings")
        if not isinstance(weight, (int, float)):
            raise InputValidationError("Asset weights must be numeric")
        if weight < 0 or weight > 1:
            raise InputValidationError("Asset weights must be between 0 and 1")
    total_weight = sum(portfolio.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise InputValidationError("Sum of portfolio weights must be 1")

def validate_distribution(distribution):
    required_methods = ['sample']
    for method in required_methods:
        if not hasattr(distribution, method):
            raise InputValidationError(f"Distribution must implement method {method}")

def validate_simulation_parameters(params):
    if not isinstance(params, dict):
        raise InputValidationError("Parameters must be a dictionary")
    if 'num_simulations' in params:
        num_sim = params['num_simulations']
        if not isinstance(num_sim, int) or num_sim <= 0:
            raise InputValidationError("num_simulations must be a positive integer")
    if 'variance_reduction' in params:
        vr = params['variance_reduction']
        allowed_methods = [None, 'importance_sampling', 'stratified_sampling', 'quasi_monte_carlo']
        if vr not in allowed_methods:
            raise InputValidationError(f"variance_reduction must be one of {allowed_methods}")
