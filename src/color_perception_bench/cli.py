"""Interactive CLI for the color perception benchmark."""

import asyncio
import sys

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from color_perception_bench.benchmark import (
    plot_model_analysis,
    print_results_table,
    run_benchmark,
)
from color_perception_bench.cache import (
    get_cache_info,
    invalidate_cache,
    list_cached_models,
    load_embeddings,
)
from color_perception_bench.registry import (
    VALID_BATCH_SIZES,
    add_model,
    create_default_local_model,
    get_model_config,
    list_models,
    remove_model,
    update_model_batch_size,
)

console = Console()

# Custom style for questionary
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ]
)


def print_header():
    """Print the application header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Color Perception Benchmark[/bold cyan]\n"
            "[dim]Multi-model embedding alignment evaluation[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def show_main_menu() -> str:
    """Show the main menu and return the selected option."""
    choices = [
        "🚀 Run Benchmark",
        "📋 View Last Results",
        "📊 Generate Plots",
        "⚙️  Manage Models",
        "🗑️  Clear Cache",
        "❌ Exit",
    ]

    return questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style,
    ).ask()


def show_model_menu() -> str:
    """Show the model management menu."""
    choices = [
        "📝 List Models",
        "🔧 Edit Batch Size",
        "➕ Add Model",
        "🗑️  Remove Model",
        "⬅️  Back to Main Menu",
    ]

    return questionary.select(
        "Model Management:",
        choices=choices,
        style=custom_style,
    ).ask()


def list_models_ui():
    """Display all registered models in a table."""
    models = list_models()

    if not models:
        console.print("[yellow]No models registered yet.[/yellow]")
        console.print("Use 'Add Model' to register your first model.")
        return

    table = Table(title="Registered Models")
    table.add_column("Name", style="cyan bold")
    table.add_column("Type", style="dim")
    table.add_column("Base URL")
    table.add_column("Text Endpoint")
    table.add_column("Image Endpoint")
    table.add_column("Batch Size", justify="right")
    table.add_column("Cached", justify="center")

    cached_models = list_cached_models()

    for name in models:
        config = get_model_config(name)
        if config:
            batch_size = config.get("user_batch_size", "auto")
            is_cached = "✓" if any(name.lower() in c for c in cached_models) else ""

            table.add_row(
                name,
                config.get("provider_type", "local"),
                config.get("base_url", ""),
                config.get("text_endpoint", {}).get("path", ""),
                config.get("image_endpoint", {}).get("path", ""),
                str(batch_size),
                is_cached,
            )

    console.print(table)


def add_model_ui():
    """Interactive UI for adding a new model."""
    console.print("\n[bold]Add New Model[/bold]\n")

    name = questionary.text(
        "Model name (unique identifier):",
        validate=lambda x: len(x) > 0 or "Name is required",
        style=custom_style,
    ).ask()

    if not name:
        return

    # Check if exists
    if name in list_models():
        console.print(f"[red]Model '{name}' already exists. Remove it first.[/red]")
        return

    provider_type = questionary.select(
        "Provider type:",
        choices=["local", "openai_compatible"],
        style=custom_style,
    ).ask()

    if not provider_type:
        return

    base_url = questionary.text(
        "Base URL (e.g., http://localhost:8080 or https://api.openai.com):",
        default=(
            "http://localhost:8080"
            if provider_type == "local"
            else "https://api.openai.com"
        ),
        style=custom_style,
    ).ask()

    if not base_url:
        return

    text_endpoint = questionary.text(
        "Text embedding endpoint path:",
        default="/txt/embed" if provider_type == "local" else "/v1/embeddings",
        style=custom_style,
    ).ask()

    if not text_endpoint:
        return

    image_endpoint = questionary.text(
        "Image embedding endpoint path:",
        default="/img/embed" if provider_type == "local" else "/v1/embeddings",
        style=custom_style,
    ).ask()

    if not image_endpoint:
        return

    # API key for openai_compatible
    api_key_env_var = None
    if provider_type == "openai_compatible":
        api_key_env_var = questionary.text(
            "Environment variable for API key (e.g., OPENAI_API_KEY):",
            default="OPENAI_API_KEY",
            style=custom_style,
        ).ask()

    # Advanced options
    show_advanced = questionary.confirm(
        "Configure advanced options?",
        default=False,
        style=custom_style,
    ).ask()

    text_input_field = "input"
    text_output_field = "embedding"
    image_input_field = "input"
    image_output_field = "embedding"
    batch_size = None

    if show_advanced:
        text_input_field = (
            questionary.text(
                "Text input field name:",
                default="input",
                style=custom_style,
            ).ask()
            or "input"
        )

        text_output_field = (
            questionary.text(
                "Text output field name:",
                default="embedding",
                style=custom_style,
            ).ask()
            or "embedding"
        )

        image_input_field = (
            questionary.text(
                "Image input field name:",
                default="input",
                style=custom_style,
            ).ask()
            or "input"
        )

        image_output_field = (
            questionary.text(
                "Image output field name:",
                default="embedding",
                style=custom_style,
            ).ask()
            or "embedding"
        )

        set_batch = questionary.confirm(
            "Set manual batch size? (otherwise auto-detect)",
            default=False,
            style=custom_style,
        ).ask()

        if set_batch:
            batch_choice = questionary.select(
                "Batch size:",
                choices=[str(s) for s in VALID_BATCH_SIZES],
                style=custom_style,
            ).ask()
            if batch_choice:
                batch_size = int(batch_choice)

    try:
        add_model(
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            text_endpoint=text_endpoint,
            image_endpoint=image_endpoint,
            api_key_env_var=api_key_env_var,
            text_input_field=text_input_field,
            text_output_field=text_output_field,
            image_input_field=image_input_field,
            image_output_field=image_output_field,
            batch_size=batch_size,
        )
        console.print(f"[green]✓[/green] Model '{name}' added successfully!")
    except Exception as e:
        console.print(f"[red]Error adding model:[/red] {e}")


def remove_model_ui():
    """Interactive UI for removing a model."""
    models = list_models()

    if not models:
        console.print("[yellow]No models to remove.[/yellow]")
        return

    model = questionary.select(
        "Select model to remove:",
        choices=models + ["⬅️  Cancel"],
        style=custom_style,
    ).ask()

    if not model or "Cancel" in model:
        return

    confirm = questionary.confirm(
        f"Are you sure you want to remove '{model}'?",
        default=False,
        style=custom_style,
    ).ask()

    if confirm:
        if remove_model(model):
            console.print(f"[green]✓[/green] Model '{model}' removed.")
        else:
            console.print("[red]Failed to remove model.[/red]")


def edit_batch_size_ui():
    """Interactive UI for editing batch size."""
    models = list_models()

    if not models:
        console.print("[yellow]No models registered.[/yellow]")
        return

    model = questionary.select(
        "Select model:",
        choices=models + ["⬅️  Cancel"],
        style=custom_style,
    ).ask()

    if not model or "Cancel" in model:
        return

    config = get_model_config(model)
    if not config:
        console.print(f"[red]Model '{model}' not found.[/red]")
        return
    current = config.get("user_batch_size", "auto")
    console.print(f"Current batch size: [cyan]{current}[/cyan]")

    choices = ["auto (discover from schema)"] + [str(s) for s in VALID_BATCH_SIZES]
    new_size = questionary.select(
        "New batch size:",
        choices=choices,
        style=custom_style,
    ).ask()

    if not new_size:
        return

    try:
        if "auto" in new_size:
            update_model_batch_size(model, None)
        else:
            update_model_batch_size(model, int(new_size))
        console.print("[green]✓[/green] Batch size updated.")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


def run_benchmark_ui():
    """Interactive UI for running benchmarks."""
    models = list_models()

    if not models:
        console.print("[yellow]No models registered.[/yellow]")
        console.print("Add a model first using 'Manage Models' > 'Add Model'.")
        return

    selected = questionary.checkbox(
        "Select models to benchmark:",
        choices=models,
        style=custom_style,
    ).ask()

    if not selected:
        console.print("[yellow]No models selected.[/yellow]")
        return

    force_refresh = questionary.confirm(
        "Force refresh (ignore cache)?",
        default=False,
        style=custom_style,
    ).ask()

    console.print()
    asyncio.run(run_benchmark(selected, force_refresh=force_refresh))


def clear_cache_ui():
    """Interactive UI for clearing cache."""
    cached = list_cached_models()

    if not cached:
        console.print("[yellow]No cached data found.[/yellow]")
        return

    # Build choices with cache info
    choices = []
    for name in cached:
        info = get_cache_info(name)
        if info:
            size = info.get("size_mb", 0)
            choices.append(f"{name} ({size} MB)")
        else:
            choices.append(name)

    choices.append("🗑️  Clear ALL")
    choices.append("⬅️  Cancel")

    selected = questionary.select(
        "Select cache to clear:",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or "Cancel" in selected:
        return

    if "Clear ALL" in selected:
        confirm = questionary.confirm(
            "Clear ALL cached data?",
            default=False,
            style=custom_style,
        ).ask()

        if confirm:
            from color_perception_bench.cache import clear_all_caches

            count = clear_all_caches()
            console.print(f"[green]✓[/green] Cleared {count} cache files.")
    else:
        # Extract model name from choice (remove size info)
        model_name = selected.split(" (")[0]
        if invalidate_cache(model_name):
            console.print(f"[green]✓[/green] Cache cleared for '{model_name}'.")
        else:
            console.print("[red]Failed to clear cache.[/red]")


def handle_model_menu():
    """Handle the model management submenu."""
    while True:
        console.print()
        choice = show_model_menu()

        if not choice or "Back" in choice:
            break
        elif "Add" in choice:
            add_model_ui()
        elif "List" in choice:
            list_models_ui()
        elif "Edit" in choice:
            edit_batch_size_ui()
        elif "Remove" in choice:
            remove_model_ui()


def generate_plots_ui():
    """Generate correlation plots for cached models."""
    console.print()
    console.rule("[bold cyan]Generate Correlation Plots[/bold cyan]")
    console.print()

    # Get list of cached models
    cached_models = list_cached_models()

    if not cached_models:
        console.print("[yellow]No cached models found. Run a benchmark first.[/yellow]")
        return

    console.print(f"Found {len(cached_models)} cached model(s)")
    console.print()

    # Let user select which models to plot
    selected = questionary.checkbox(
        "Select models to generate plots for:",
        choices=cached_models,
        style=custom_style,
    ).ask()

    if not selected:
        console.print("[dim]No models selected.[/dim]")
        return

    console.print()
    console.print(f"Generating plots for {len(selected)} model(s)...")
    console.print()

    # Generate plots for each selected model
    for model_name in selected:
        try:
            data = load_embeddings(model_name)
            if data is None:
                console.print(f"[red]✗[/red] Failed to load cache for {model_name}")
                continue

            # Check if cache is complete
            sample = next(iter(data.values()))
            if (
                sample.get("text_embedding") is None
                or sample.get("image_embedding") is None
            ):
                console.print(
                    f"[yellow]⚠[/yellow] Incomplete cache for {model_name}, skipping"
                )
                continue

            plot_model_analysis(model_name, data)

        except Exception as e:
            console.print(f"[red]✗[/red] Error generating plot for {model_name}: {e}")

    console.print()
    console.print("[green]✓[/green] Plot generation complete!")


def main():
    """Main entry point for the CLI."""
    print_header()

    # Ensure default local model exists
    create_default_local_model()

    while True:
        choice = show_main_menu()

        if not choice or "Exit" in choice:
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)
        elif "Run" in choice:
            run_benchmark_ui()
        elif "View" in choice:
            console.print()
            print_results_table()
        elif "Plot" in choice:
            generate_plots_ui()
        elif "Manage" in choice:
            handle_model_menu()
        elif "Clear" in choice:
            clear_cache_ui()


if __name__ == "__main__":
    main()
