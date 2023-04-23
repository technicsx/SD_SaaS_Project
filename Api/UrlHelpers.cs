using System.Runtime.CompilerServices;

namespace Api
{
    public static class UrlHelpers
    {
        public static FormattableString EncodeUrlParameters(FormattableString url)
        {
            return FormattableStringFactory.Create(
                url.Format,
                url.GetArguments()
                    .Select(a => Uri.EscapeDataString(a?.ToString() ?? ""))
                    .ToArray());
        }
    }
}