# Armchair Analyst Vision

Armchair Analyst is a desktop application for analyzing blockchain market data across centralized exchanges, decentralized exchanges, and other trading venues.

The application will continuously monitor a broad set of data sources, aggregate the resulting market and chain activity, and run analysis over that data to surface useful signals for the user.

## Core Direction

- Build the core data collection, aggregation, and analysis logic in Rust.
- Support both CEX and DEX data sources, with room for additional exchange-like venues over time.
- Start with a Windows desktop shell implemented with WPF.
- Support automatic application updates.
- Support background data stream monitoring and notifications even when the main application window is closed.
- Treat the Windows background agent as the production application host and the WPF UI as a thin user interaction layer.
- Use an installer or package for production Windows distribution so the background agent, updates, notifications, shortcuts, and uninstall behavior are registered correctly.
- Preserve room for additional OS-specific shells in the future.
- Keep architecture details in [architecture.md](architecture.md).

## Product Intent

The goal is to give users a local desktop tool that can observe fragmented blockchain markets, combine data from many sources, and provide timely analysis without tying the core system to a single operating system or UI technology.

On Windows, the app should feel like a persistent local market monitor: users can close the main window while background monitoring continues in their logged-in session and important findings can still appear as notifications.
