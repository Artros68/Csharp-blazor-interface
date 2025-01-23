using System.IO;
using GreenScale.Data;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// New path for the wwwroot folder
var options = new WebApplicationOptions
{
    WebRootPath = Path.Combine(
        Directory.GetCurrentDirectory(),
        "GreenScale.Presentation",
        "wwwroot"
    ),
};

var builder = WebApplication.CreateBuilder(options);

// Add services to the dependency container
builder.Services.AddRazorPages(options =>
{
    options.RootDirectory = "/GreenScale.Presentation/Pages";
});
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();

var app = builder.Build();

// Configuring the HTTP pipeline
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
            Path.Combine(builder.Environment.ContentRootPath, "GreenScale.Presentation", "wwwroot")
        ),
        RequestPath = "",
    }
);
app.UseRouting();

// Configuring Blazor endpoints
app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

// Start application
app.Run();
