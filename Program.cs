using System.IO;
using Csharp_blazor_interface.Data;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

// Ajouter les services au conteneur de dépendances
builder.Services.AddRazorPages(options =>
{
    options.RootDirectory = "/Csharp-blazor-interface.Presentation/Pages";
});
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();

var app = builder.Build();

// Configurer le pipeline HTTP
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles(
    new StaticFileOptions
    {
        FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(
            Path.Combine(
                builder.Environment.ContentRootPath,
                "Csharp-blazor-interface.Presentation",
                "wwwroot"
            )
        ),
        RequestPath = "",
    }
);
app.UseRouting();

// Configuration des endpoints Blazor
app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

// Démarrer l'application
app.Run();
