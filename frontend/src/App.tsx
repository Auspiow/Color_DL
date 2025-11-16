import { useEffect, useState } from "react";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Box,
  CircularProgress,
  createTheme,
  ThemeProvider,
} from "@mui/material";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#4285F4" },
    secondary: { main: "#EA4335" },
    background: { default: "#f8f9fa", paper: "#ffffff" },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      fontSize: "2.125rem",
    },
    h6: {
      fontWeight: 500,
      fontSize: "1rem",
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: "0 1px 3px rgba(60,64,67,.3), 0 4px 8px rgba(60,64,67,.15)",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            boxShadow: "0 1px 3px rgba(60,64,67,.3), 0 8px 16px rgba(60,64,67,.25)",
            transform: "translateY(-4px)",
          },
        },
      },
    },
  },
});

export default function App() {
  const [files, setFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/api/plots")
      .then((res) => res.json())
      .then((data) => {
        setFiles(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ backgroundColor: "#f8f9fa", minHeight: "100vh" }}>
        <Container maxWidth="lg" sx={{ paddingY: 4 }}>
          {/* Header Section */}
          <Box sx={{ marginBottom: 6 }}>
            <Typography 
              variant="h4" 
              gutterBottom 
              sx={{ 
                color: "#202124",
                marginBottom: 1,
              }}
            >
              🎨 色感知模型 — 可视化报告
            </Typography>

            <Typography 
              variant="body1" 
              sx={{ 
                color: "#5f6368",
                fontSize: "1.0625rem",
                lineHeight: 1.6,
                maxWidth: "700px",
              }}
            >
              基于深度学习的色差预测模型，展示模型性能可视化
            </Typography>
          </Box>

          {/* Loading State */}
          {loading && (
            <Box sx={{ display: "flex", justifyContent: "center", paddingY: 8 }}>
              <CircularProgress />
            </Box>
          )}

          {/* Cards Grid */}
          {!loading && (
            <Grid container spacing={3}>
              {files.map((file) => (
                <Grid key={file} sx={{ xs: 12, sm: 6, md: 4, flex: "100%", "@media (min-width: 600px)": { flex: "50%" }, "@media (min-width: 960px)": { flex: "33.333%" } }}>
                  <Card>
                    <CardMedia
                      component="img"
                      height="280"
                      image={`http://localhost:8000/plots/${file}`}
                      alt={file.replace(".png", "")}
                      sx={{ 
                        objectFit: "contain",
                        padding: 2,
                        backgroundColor: "#fafafa",
                      }}
                    />
                    <CardContent sx={{ paddingTop: 2 }}>
                      <Typography 
                        variant="h6" 
                        sx={{ 
                          color: "#202124",
                          fontSize: "0.95rem",
                          fontWeight: 500,
                          textTransform: "capitalize",
                        }}
                      >
                        {file.replace(".png", "").replaceAll("_", " ")}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}

          {/* Empty State */}
          {!loading && files.length === 0 && (
            <Box sx={{ textAlign: "center", paddingY: 8 }}>
              <Typography color="text.secondary">
                没有找到可视化文件
              </Typography>
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
