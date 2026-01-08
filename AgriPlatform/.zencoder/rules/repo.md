---
description: Repository Information Overview
alwaysApply: true
---

# AgriPlatform Information

## Summary

AgriPlatform is a full-stack web application for agricultural management, allowing users to create, view, and manage farm data. The platform includes features for mapping farm boundaries, tracking crop information, and managing planting/harvest dates.

## Structure

The repository is organized as a monorepo with two main components:

- **Frontend**: React application built with TypeScript, Vite, and TailwindCSS
- **Backend**: Express.js API with MongoDB database

## Projects

### Frontend

**Configuration File**: package.json

#### Language & Runtime

**Language**: TypeScript
**Version**: TypeScript ~5.8.3
**Build System**: Vite 7.1.2
**Package Manager**: npm

#### Dependencies

**Main Dependencies**:

- React 19.1.1
- React Router DOM 7.8.2
- Leaflet 1.9.4 (for mapping)
- Mapbox-GL 3.14.0
- Zustand 5.0.8 (state management)
- Lucide-React 0.542.0 (icons)

**Development Dependencies**:

- ESLint 9.35.0
- Prettier 3.6.2
- TailwindCSS 3.4.17
- TypeScript 5.8.3
- Vite 7.1.2

#### Build & Installation

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
```

#### Main Files

**Entry Point**: src/main.tsx
**Router**: src/router.tsx
**Main Components**:

- src/App.tsx
- src/pages/\* (Dashboard, FarmDetail, CreateFarm, etc.)
- src/components/\* (UI components including map components)

**State Management**:

- src/contexts/AuthContext.tsx (Authentication)
- src/stores/\* (Zustand stores)

### Backend

**Configuration File**: package.json

#### Language & Runtime

**Language**: JavaScript (Node.js)
**Framework**: Express 5.1.0
**Database**: MongoDB with Mongoose 8.18.1

#### Dependencies

**Main Dependencies**:

- Express 5.1.0
- Mongoose 8.18.1
- JWT 9.0.2
- Bcrypt 6.0.0
- Dotenv 17.2.2
- Cors 2.8.5

**Development Dependencies**:

- Nodemon 3.1.10

#### Build & Installation

```bash
# Install dependencies
npm install

# Development server with auto-reload
npm run dev

# Production server
npm run start
```

#### Main Files

**Entry Point**: server.js
**API Setup**: app.js
**Routes**:

- routes/user.route.js
- routes/farm.route.js

**Models**:

- models/user.model.js
- models/farm.model.js

**Controllers**:

- controllers/user.controller.js
- controllers/farm.controller.js

## CI/CD

**Workflow**: GitHub Actions (.github/workflows/main.yml)
**Triggers**: Push to main branch, Pull requests to main branch
**Steps**:

- Checkout code
- Run tests (placeholder)
- Send Discord notification
