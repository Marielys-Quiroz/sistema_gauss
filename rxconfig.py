import reflex as rx

config = rx.Config(
    app_name="sistema_gauss",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)