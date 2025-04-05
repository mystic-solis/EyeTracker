import click


def validate_percentage(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter(f"{param.name} должен быть от 0 до 1, а не {value}")
    return value